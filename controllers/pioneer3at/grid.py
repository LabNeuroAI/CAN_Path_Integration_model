import numpy as np

class Grid():
    def __init__(self):
        
        self.mm = 20
        self.nn = 20
        self.TAO = 0.9
        self.II = 0.3
        self.SIGMA = 0.24
        self.SIGMA2 = self.SIGMA**2
        self.TT = 0.05
        self.grid_gain = [0.04,0.05,0.06,0.07,0.08]
        self.grid_layers = len(self.grid_gain)  
        self.grid_activity = np.random.uniform(0,1,(self.mm,self.nn,self.grid_layers))  
        self.distTri = self.buildTopology(self.mm,self.nn)


    def update(self, speedVector):

        self.speedVector = speedVector
        
        grid_ActTemp = []
        for jj in range(0,self.grid_layers):
            rrr = self.grid_gain[jj]*np.exp(1j*0)
            matWeights = self.updateWeight(self.distTri,rrr)
            activityVect = np.ravel(self.grid_activity[:,:,jj])
            activityVect = self.Bfunc(activityVect, matWeights)
            activityTemp = activityVect.reshape(self.mm,self.nn)
            activityTemp += self.TAO *( activityTemp/np.mean(activityTemp) - activityTemp)
            activityTemp[activityTemp<0] = 0

            self.grid_activity[:,:,jj] = (activityTemp-np.min(activityTemp))/(  np.max(activityTemp)-np.min(activityTemp)) * 30  ##Eq 2
                        

    def buildTopology(self,mm,nn):  # Build connectivity matrix     ### Eq 4
        mmm = (np.arange(mm)+(0.5/mm))/mm
        nnn = ((np.arange(nn)+(0.5/nn))/nn)*np.sqrt(3)/2
        xx,yy = np.meshgrid(mmm, nnn)
        posv = xx+1j * yy
        Sdist = [ 0+1j*0, -0.5+1j*np.sqrt(3)/2, -0.5+1j*(-np.sqrt(3)/2), 0.5+1j*np.sqrt(3)/2, 0.5+1j*(-np.sqrt(3)/2), -1+1j*0, 1+1j*0]      
        xx,yy = np.meshgrid( np.ravel(posv) , np.ravel(posv) )
        distmat = xx-yy
        for ii in range(len(Sdist)):
            aaa1 = abs(distmat)
            rrr = xx-yy + Sdist[ii]
            aaa2 = abs(rrr)
            iii = np.where(aaa2<aaa1)
            distmat[iii] = rrr[iii]
        return distmat.transpose()

    def updateWeight(self,topology,rrr): # Slight update on weights based on speed vector.
        matWeights = self.II * np.exp((-abs(topology-rrr*self.speedVector)**2)/self.SIGMA2) - self.TT   ## Eq 3
        return matWeights

    def Bfunc(self,activity, matWeights):  ## Eq 1
        activity += np.dot(activity,matWeights)
        return activity