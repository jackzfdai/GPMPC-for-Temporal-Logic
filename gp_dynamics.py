import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from casadi import *

#Trains a 1-dimensional GP residual 
class gpResidual: 
    def __init__(self, stateDim, inputDim):
        self.stateDim = stateDim
        self.inputDim = inputDim
        
        self.x = SX.sym('x', self.stateDim)
        self.xCovar = SX.sym('xCovar', self.stateDim)
        self.u = SX.sym('u', self.inputDim)

        self.l = 0
        self.sigma = 0
        self.noise = 0
        self.KzzPrior = torch.tensor([])
        self.inputTrainingData = torch.tensor([])

    def train(self, trainingIt, trainingInputData, trainingOutputData):
        self.trainingIt = trainingIt
        self.inputTrainingData = trainingInputData
        self.outputTrainingData = trainingOutputData

        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(self.inputTrainingData, self.outputTrainingData, self.likelihood)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.trainingIt):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.inputTrainingData)
            # Calc loss and backprop gradients
            loss = -mll(output, self.outputTrainingData)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.6f  output scale: %.10f   noise: %.10f' % (
                i + 1, self.trainingIt, loss.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                self.model.covar_module.outputscale.item(),
                self.model.likelihood.noise.item()
            ))
            optimizer.step()

        self.model.eval()
        self.likelihood.eval()

        test_out = self.model.covar_module(self.inputTrainingData)
        test_out_dense = test_out.to_dense()
        # print(test_out_dense)
        # print(self.model.covar_module.base_kernel.lengthscale)
        # print(self.model.covar_module.outputscale)

        self.KzzPrior = self.model.covar_module(self.inputTrainingData).to_dense()
        # print(len(self.inputTrainingData))
        self.dMu = torch.matmul(torch.linalg.inv(self.KzzPrior + self.model.likelihood.noise.item()*torch.eye(self.KzzPrior.size()[0])), self.outputTrainingData)
        self.dSigma = torch.linalg.inv(self.KzzPrior + self.model.likelihood.noise.item()*torch.eye(self.KzzPrior.size()[0]))

        self.l = self.model.covar_module.base_kernel.lengthscale
        self.sigma = self.model.covar_module.outputscale

        print("l: ", self.l.item())
        print("sigma (scale): ", self.sigma.item())
        print("sigma (noise): ", self.model.likelihood.noise.item())

    def parametersFromFile(self, modelFile):
        modelFile.seek(0)

        line1 = modelFile.readline()
        self.l = torch.tensor(float(line1.split("\n")[0]))
        modelFile.readline()

        line2 = modelFile.readline()
        self.sigma = torch.tensor(float(line2.split("\n")[0]))
        modelFile.readline()

        line3 = modelFile.readline()
        self.noise = torch.tensor(float(line3.split("\n")[0]))
        modelFile.readline()

        trainingDataLines = []
        line = modelFile.readline()
        while not(line == "~\n"):
            lineElements = line.split(",")
            lineElementsFloat = []
            for lineElement in lineElements[:-1]:
                lineElementsFloat += [float(lineElement)]
            
            trainingDataLine = np.array(lineElementsFloat)
            trainingDataLines += [trainingDataLine]
            line = modelFile.readline()

        self.inputTrainingData = torch.tensor(np.vstack(trainingDataLines))

        trainingOutputDataLines = []
        line = modelFile.readline()
        while not(line == "~\n"):
            lineElements = line.split(",")
            trainingOutputDataLines += [float(lineElements[0])]
            line = modelFile.readline()

        self.outputTrainingData = torch.tensor(np.hstack(trainingOutputDataLines))

        KzzPriorLines = []
        line = modelFile.readline()
        while not(line == "~\n"):
            lineElements = line.split(",")
            lineElementsConcatenate = []
            for lineElement in lineElements[:-1]:
                lineElementsConcatenate += [float(lineElement)]
            
            KzzPriorLine = np.hstack(lineElementsConcatenate)
            KzzPriorLines += [KzzPriorLine]
            line = modelFile.readline()

        self.KzzPrior = torch.tensor(np.vstack(KzzPriorLines))

        self.x = SX.sym('x', self.stateDim)
        self.xCovar = SX.sym('xCovar', self.stateDim)
        self.u = SX.sym('u', self.inputDim)

        self.dMu = torch.matmul(torch.linalg.inv(self.KzzPrior + self.noise.item()*torch.eye(self.KzzPrior.size()[0])), self.outputTrainingData)
        self.dSigma = torch.linalg.inv(self.KzzPrior + self.noise.item()*torch.eye(self.KzzPrior.size()[0]))


    def getResidualFunction(self):
        # print(len(self.inputTrainingData))
        residualMean = self.sigma.item()*exp((-1/(2*self.l.item()**2))*(sum2((repmat(horzcat(transpose(self.x), transpose(self.u)), len(self.inputTrainingData), 1) - DM(self.inputTrainingData.detach().numpy()))**2).T)) @ DM(self.dMu.detach().numpy())
        residualCovariance = self.sigma.item() - (self.sigma.item()*exp((-1/(2*(self.l.item()**2)))*(sum2((repmat(horzcat(transpose(self.x), transpose(self.u)), len(self.inputTrainingData), 1) - DM(self.inputTrainingData.detach().numpy()))**2).T))) @ DM(self.dSigma.detach().numpy()) @ (self.sigma.item()*exp((-1/(2*(self.l.item()**2)))*(sum2((repmat(horzcat(transpose(self.x), transpose(self.u)), len(self.inputTrainingData), 1) - DM(self.inputTrainingData.detach().numpy()))**2).T))).T
        jacobianResidualMean = horzcat(jacobian(residualMean, self.x), jacobian(residualMean, self.u))
        jacobianResidualCovariance = horzcat(jacobian(residualCovariance, self.x), jacobian(residualCovariance, self.u))
        # print(jacobianResidualCovariance[0])
        # testx = DM([28, 0.6, 7, 1])
        # testu = DM([1, 0])
        # print(jacobianResidualCovariance)
        G = Function('G', [self.x, self.u], [residualMean, jacobianResidualMean, residualCovariance, jacobianResidualCovariance], ['x', 'u'], ['mean', 'jacmean', 'covar', 'jaccovar'])
        # print(len(self.inputTrainingData))
        # testcovar = G(x=testx, u=testu)['jaccovar']
        # print(testcovar)
        # print("____")
        return G
    
    def getStateCovarVar(self):
        return self.xCovar
    
    def testResidual(self, inputData, expectedOutputData):
        observed_pred = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(inputData))

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(inputData[:,0].numpy(), expectedOutputData.numpy(), 'k*')
            # ax.plot(inputData[:,0].numpy(), expectedOutputData.numpy(), 'k+')
            # Plot predictive means as blue line
            ax.plot(inputData[:,0].numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(inputData[:,0].numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_ylim([-1, 1])
            ax.legend(['Observed Data','Mean', 'Confidence'])
            plt.show()
        return observed_pred.mean.numpy()
    
    def getPrediction(self, inputData):
        observed_pred = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(inputData))

        return observed_pred
    
    def writeModelTo(self, modelFile):
        modelFile.write(str(self.l.item()) + "\n")
        modelFile.write("~\n")
        modelFile.write(str(self.sigma.item()) + "\n")
        modelFile.write("~\n")
        modelFile.write(str(self.model.likelihood.noise.item()) + "\n")
        modelFile.write("~\n")
        
        trainingData = self.inputTrainingData.detach().numpy()
        for i in range(trainingData.shape[0]):
            for j in range(trainingData.shape[1]):
                modelFile.write(str(trainingData[i, j]) + ",")
            modelFile.write("\n")
        modelFile.write("~\n")

        trainingOutputData = self.outputTrainingData.detach().numpy()
        for output in trainingOutputData:
            modelFile.write(str(output.item()) + ",")
            modelFile.write("\n")
        modelFile.write("~\n")

        KzzPrior = self.KzzPrior.detach().numpy()
        for i in range(KzzPrior.shape[0]):
            for j in range(KzzPrior.shape[1]):
                modelFile.write(str(KzzPrior[i, j]) + ",")
            modelFile.write("\n")

        modelFile.write("~\n")
    
    # def getResidualCovarianceFunctionse


        

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)