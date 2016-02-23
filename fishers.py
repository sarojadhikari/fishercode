"""
.. module:: fishers
  :ynopsis: collection of fisher matrices; useful for combined visualization
"""
import matplotlib.pyplot as plt

class FisherMatrices(object):
    """
    This defines a collection of fisher matrices (Fisher class)
    """
    def __init__(self, fisherlist=[]):
        self.fms = fisherlist
        self.Nfms = len(self.fms)
        self.clrs = ["g", "b", "r", "black", "o"]
        self.alpha = 0.2

    def add_fisher_matrix(self, fm):
        self.fms.append(fm)

    add_fm = add_fisher_matrix

    def plot_error_matrix(self, params, nstd=1, nbinsx=6, nbinsy=6):
        """ plot a matrix of fisher forecast error ellipses for all the
        fisher matrices in the self.fms list
        """
        fac = len(params)-1
        #plt.close('all')
        plt.ticklabel_format(style='sci', axis='both', scilimits=(-3,3))

        f, allaxes = plt.subplots(fac, fac, sharex="col", sharey="row")
        for i in range(fac):
            for j in range(fac):
                if (j>i):
                    allaxes[i][j].axis('off')

        if fac<2:
            fmcnt=0
            for fm in self.fms:
                errorellipse, xyaxes=fm.error_ellipse(fm.params[0],fm.params[1], nstd=nstd, clr=self.clrs[fmcnt], alpha=self.alpha)
                #ere2, xyaxes2 = self.error_ellipse(fm.params[0], fm.params[1], nstd=2, clr="r", alpha=0.1)
                #allaxes.add_artist(ere2)
                allaxes.add_artist(errorellipse)
                allaxes.axis(xyaxes)
                fmcnt = fmcnt + 1

            allaxes.set_xlabel(self.fms[0].param_names[0], fontsize=14)
            allaxes.set_ylabel(self.fms[0].param_names[1], fontsize=14)

        else:
            for j in params:
                for i in params:
                    if (j>i):
                        fmcnt=0
                        for fm in self.fms:
                            errorellipse, xyaxes=fm.error_ellipse(i,j, nstd=nstd, clr=self.clrs[fmcnt], alpha=self.alpha)
                            jp=i
                            ip=j-1
                            axis=allaxes[ip][jp]
                            axis.add_artist(errorellipse)
                            if fmcnt == 0:
                                axis.axis(xyaxes)

                            fmcnt=fmcnt+1

                        axis.ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
                        axis.locator_params(axis='x', nbins=nbinsx)
                        axis.locator_params(axis='y', nbins=nbinsy)
                        axis.set_xlabel(self.fms[0].param_names[i], fontsize=14)
                        axis.set_ylabel(self.fms[0].param_names[j], fontsize=14)
                        #allaxes[jp][ip].set_title(str(jp)+","+str(ip))
