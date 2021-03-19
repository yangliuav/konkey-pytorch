# from losses import mse
import numpy as np

class AECLoss():
    r"""Calculate all AEC objective functions in one class.
    
    Parameters:
        names: the name list of objective functions.
               These objective functions include:
               mse : Mean Square Error 
               pmsqe : Perceptual Metric for Speech Quality Evaluation
    """


    def __init__(names = []):
        self.names = names

        self.lossList = []
        for name in names:
            if name == "mse":
                from losses import mse
                loss = SingleSrcMSE().cuda()
                lossList.append(loss)
            if name == "pmsqe"
                from losses import pmsqe
                loss = SingleSrcPMSQE().cuda()
                lossList.append(loss)
        
    def forward(est_targets, targets):
    """
    Shape:
        - est_targets : :math:`(batch, nchan, nsample)`.
        - targets: :math:`(batch, nchan, nsample)`.

    Returns:
        :class: narray: with shape :math:`(len(names)+1)`
        the first item is the sum of losses.
        the following items are the "names" losses.

    Examples
    """
        loss = np.zeros((len(names)+1))
        index = 0
        for name in names:
            if name == "mse":
                loss[index + 1] = lossList[index].forward(est_targets, targets)
            if name == "pmsqe":
                
                est_spec = transforms.mag(self.stft(ests[0].unsqueeze(1)))
                ref_spec  = transforms.mag(self.stft(ests[1].unsqueeze(1)))
                pmsqe = self.pmsqe_func(est_spec, ref_spec)
                loss_pmsqe = torch.mean(pmsqe)


            index = index + 1
