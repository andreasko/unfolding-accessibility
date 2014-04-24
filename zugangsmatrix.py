'''
Die Klasse erstellt eine Zugangsmatrix und wertet sie aus.
Funktionen:
    1. getMatrix
            Eingabe:    Liste von Adjazenzmatrizen A(n) die den aktiven Teil des Netzwerks zum Zeitpunkt n beschreibt,
                        Infektioese Periode
            Ausgabe:    Zugangsmatrix
'''
import numpy as np
from scipy import sparse as sp
from einlesen import readdata
from networkx.convert import to_scipy_sparse_matrix
from networkx.generators.random_graphs import fast_gnp_random_graph

class Zugangsmatrix:
    def __init__(self):
        self.zm = []

    def load(self, file):
        results = np.load(file)
        self.graph = results['graph']
        self.size = results['size']
        self.prob = results['prob']
        self.runtime = results['runtime']
        self.mode = results['mode']
        self.memory = results['memory']
        indices = results['indices']
        indptr = results['indptr']
        
        self.zm = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.runtime+1)]
        if len(indices) == len(indptr):
            self.runtime = len(indptr)-1
            for ii in range(self.runtime+1):
                data = np.ones((1,len(indices[ii])),dtype=np.int8)[0]
                self.zm[ii] = sp.csr_matrix((data,indices[ii],indptr[ii]), shape=(self.size,self.size))
            return
        else:
            return 'there must be as many matrices in indices as in indptr'

    def save(self,path="/users/stud/koher/workspace/master/", fname = "zugangsmatrix"):
        self.path = str(path)
        self.fname = str(fname)
        file = self.path + str(self.fname)
        indices = [self.zm[ii].indices for ii in range(self.runtime+1)]
        indptr = [self.zm[ii].indptr for ii in range(self.runtime+1)]
        np.savez(file, indices=indices, indptr=indptr, graph=self.graph, size=self.size, prob=self.prob, runtime=self.runtime, mode=self.mode, memory=self.memory)

    def SI_load(self,A):
        zm = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.runtime+1)]
        zm[0] = sp.eye(self.size,self.size,dtype=np.int8,format='csr')
        C = A[0] + sp.eye(self.size, self.size, 0, format='csr')
        for ii in range(1,self.runtime+1):
            C.data = np.int8(C.data>0)
            zm[ii] = C
            C = (A[ii-1] + sp.eye(self.size, self.size, 0, format='csr'))*C
        return zm

    def SIR_load(self,A):
        #=======================================================================
        # SIR_load() berechnet ein Ausbruchsszenario nach dem SIR-Modell.
        # Als Adjazenzmatrizen wird eine gegebene Serie unter 'graph.dat' geladen.
        #=======================================================================
        
        #=======================================================================
        # Berechnung der Leslie-Matrix fuer den ersten Zeitschritt
        #=======================================================================
        T = sp.lil_matrix((self.memory+1,self.memory+1),dtype=np.int8)
        T[0,0:-1] = 1
        T = T.tocsr()
        K = sp.eye(self.memory+1,self.memory+1,1,dtype=np.int8,format='lil').transpose()
        K[-1,-1] = 1
        M = sp.kron(K.tocsr(), sp.eye(self.size,self.size,dtype=np.int8, format='csr'),format='csr')
        L =  M + sp.kron(T,A[0],format='csr')
        C = L[:,0:self.size]
        tmp = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.memory+1)]
        C = self.clearMat(C,tmp)
        
        #=======================================================================
        # C beinhaltet die Altersklassen des Netzwerks
        # sumup ist eine Hilfsmatrix um C blockweise aufzusummieren
        # Die Zugangsmatrix zm zum Zeitpunkt 0 ist die Einheitsmatrix und
        # zum Zeitpunkt 1 gerade gleich der ersten Adjazenzmatrix und eins wenn 
        # die Infektionsperiode groesser ist als 0
        #=======================================================================
        sumup = sp.hstack(np.append([sp.eye(self.size,self.size,dtype=np.int8, format='csr') for ii in range(self.memory)],[sp.csr_matrix((self.size,self.size),dtype=np.int8)])).tocsr()
        zm = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.runtime+1)]
        zm[0] = sp.eye(self.size,self.size,dtype=np.int8, format='csr')
        if self.memory > 0:
            zm[1] = zm[0] + C[0:self.size,:]
        else:
            zm[1] = C[0:self.size,:]
        #=======================================================================
        # Die zeitabhaengige Zugangsmatrix wird berechnet
        #=======================================================================
        for ii in range(2,self.runtime+1):
            L = M + sp.kron(T,A[ii-1], format='csr')
            C = L*C
            C = self.clearMat(C, tmp)
            zm[ii] = sumup*C
        return zm

    def SIS_load(self,A):
        #=======================================================================
        # SIR_load() berechnet ein Ausbruchsszenario nach dem SIR-Modell.
        # Als Adjazenzmatrizen wird eine gegebene Serie unter 'graph.dat' geladen.
        #=======================================================================
        
        #=======================================================================
        # Berechnung der Leslie-Matrix fuer den ersten Zeitschritt
        #===================================================================
        T = sp.lil_matrix((self.memory+1,self.memory+1),dtype=np.int8)
        T[0,:] = 1
        T = T.tocsr()
        K = sp.eye(self.memory+1,self.memory+1,1,dtype=np.int8,format='csr').transpose()
        M = sp.kron(K, sp.eye(self.size,self.size,dtype=np.int8, format='csr'),format='csr')
        L =  M + sp.kron(T,A[0],format='csr')        
        #=======================================================================
        # C beinhaltet die Altersklassen des Netzwerks
        # sumup ist eine Hilfsmatrix um C blockweise aufzusummieren
        # Die Zugangsmatrix zm zum Zeitpunkt 0 ist die Einheitsmatrix und
        # zum Zeitpunkt 1 gerade gleich der ersten Adjazenzmatrix und eins wenn 
        # die Infektionsperiode groesser ist als 0
        #=======================================================================
        C = L[:,0:self.size]
        tmp = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.memory+1)]
        C = self.clearMat(C,tmp)
        sumup = sp.hstack([sp.eye(self.size,self.size,dtype=np.int8, format='csr') for ii in range(self.memory+1)]).tocsr()
        zm = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.runtime+1)]
        zm[0] = sp.eye(self.size,self.size,dtype=np.int8, format='csr')
        if self.memory > 0:
            zm[1] = zm[0] + C[0:self.size,:]
        else:
            zm[1] = C[0:self.size,:]
        #=======================================================================
        # Die zeitabhaengige Zugangsmatrix wird berechnet
        #=======================================================================
        for ii in range(2,self.runtime+1):
            L = M + sp.kron(T,A[ii-1], format='csr')
            C = L*C
            C = self.clearMat(C, tmp)
            zm[ii] = sumup*C
        return zm

    def SI_random(self):
        zm = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.runtime+1)]
        zm[0] = sp.eye(self.size,self.size,dtype=np.int8, format='csr')
        C = to_scipy_sparse_matrix(fast_gnp_random_graph(self.size,self.prob),dtype=np.int8,format='csr') + sp.eye(self.size, self.size, 0, dtype=np.int8, format='csr')
        for ii in range(1,self.runtime+1):
            zm[ii] = C
            C = (to_scipy_sparse_matrix(fast_gnp_random_graph(self.size,self.prob),dtype=np.int8,format='csr') + sp.eye(self.size, self.size, 0, dtype=np.int8, format='csr'))*C
        return zm
    
    def SIR_random(self):
        #=======================================================================
        # SIR_random() berechnet ein Ausbruchsszenario nach dem SIR-Modell.
        # Als Adjazenzmatrizen werden zu jedem Zeitschritt Erdos-Renyi-Graphen ausgewuerfelt.
        #=======================================================================
        
        #=======================================================================
        # Berechnung der Leslie-Matrix fuer den ersten Zeitschritt
        #=======================================================================
        T = sp.lil_matrix((self.memory+1,self.memory+1),dtype=np.int8)
        T[0,0:-1] = 1
        T = T.tocsr()
        K = sp.eye(self.memory+1,self.memory+1,1,dtype=np.int8,format='csr').transpose()
        K[-1,-1] = 1
        M = sp.kron(K, sp.eye(self.size,self.size,dtype=np.int8, format='csr'), format='csr')
        A = to_scipy_sparse_matrix(fast_gnp_random_graph(self.size,self.prob),dtype=np.int8,format='csr')
        L =  M + sp.kron(T,A, format='csr')
        
        #=======================================================================
        # C beinhaltet die Altersklassen des Netzwerks
        # sumup ist eine Hilfsmatrix um C blockweise aufzusummieren
        # Die Zugangsmatrix zm zum Zeitpunkt 0 ist die Einheitsmatrix und
        # zum Zeitpunkt 1 gerade gleich der ersten Adjazenzmatrix und eins wenn 
        # die Infektionsperiode groesser ist als 0
        #=======================================================================
        C = L[:,0:self.size]
        tmp = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.memory+1)]
        sumup = sp.hstack(np.append([sp.eye(self.size,self.size,dtype=np.int8, format='csr') for ii in range(self.memory)],[sp.csr_matrix((self.size,self.size),dtype=np.int8)])).tocsr()
        zm = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.runtime+1)]
        zm[0] = sp.eye(self.size,self.size,dtype=np.int8, format='csr')
        if self.memory > 0:
            zm[1] = zm[0] + C[0:self.size,:]
        else:
            zm[1] = C[0:self.size,:]
        #=======================================================================
        # Die zeitabhaengige Zugangsmatrix wird berechnet
        #=======================================================================
        for ii in range(2,self.runtime+1): #runtime > 1
            A = to_scipy_sparse_matrix(fast_gnp_random_graph(self.size,self.prob),dtype=np.int8,format='csr')
            L = M + sp.kron(T,A, format='csr')
            C = L*C
            C = self.clearMat(C, tmp)
            zm[ii] = sumup*C
        return zm

    def SIS_random(self):
        #=======================================================================
        # SIR_random() berechnet ein Ausbruchsszenario nach dem SIR-Modell.
        # Als Adjazenzmatrizen werden zu jedem Zeitschritt Erdos-Renyi-Graphen ausgewuerfelt.
        #=======================================================================
        
        #=======================================================================
        # Berechnung der Leslie-Matrix fuer den ersten Zeitschritt
        #=======================================================================
        T = sp.lil_matrix((self.memory+1,self.memory+1),dtype=np.int8)
        T[0,:] = 1
        T = T.tocsr()
        K = sp.eye(self.memory+1,self.memory+1,1,dtype=np.int8,format='csr').transpose()
        M = sp.kron(K, sp.eye(self.size,self.size,dtype=np.int8, format='csr'), format='csr')
        A = to_scipy_sparse_matrix(fast_gnp_random_graph(self.size,self.prob),dtype=np.int8,format='csr')
        L =  M + sp.kron(T,A, format='csr')
        
        #=======================================================================
        # C beinhaltet die Altersklassen des Netzwerks
        # sumup ist eine Hilfsmatrix um C blockweise aufzusummieren
        # Die Zugangsmatrix zm zum Zeitpunkt 0 ist die Einheitsmatrix und
        # zum Zeitpunkt 1 gerade gleich der ersten Adjazenzmatrix und eins wenn 
        # die Infektionsperiode groesser ist als 0
        #=======================================================================
        C = L[:,0:self.size]
        tmp = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.memory+1)]
        sumup = sp.hstack([sp.eye(self.size,self.size,dtype=np.int8, format='csr') for ii in range(self.memory+1)]).tocsr()
        zm = [sp.csr_matrix((self.size,self.size),dtype=np.int8) for ii in range(self.runtime+1)]
        zm[0] = sp.eye(self.size,self.size,dtype=np.int8, format='csr')
        if self.memory > 0:
            zm[1] = zm[0] + C[0:self.size,:]
        else:
            zm[1] = C[0:self.size,:]
        
        #=======================================================================
        # Die zeitabhaengige Zugangsmatrix wird berechnet
        #=======================================================================
        for ii in range(2,self.runtime+1): #runtime > 1
            A = to_scipy_sparse_matrix(fast_gnp_random_graph(self.size,self.prob),dtype=np.int8,format='csr')
            L = M + sp.kron(T,A, format='csr')
            C = L*C
            C = self.clearMat(C, tmp)
            zm[ii] = sumup*C
        return zm

    def clearMat(self,L,tmp):
        
        cum = L[self.memory*self.size:(self.memory+1)*self.size,:]
        tmp[-1] = cum
        for ii in range(self.memory,0,-1):
            tmp[ii-1] = L[(ii-1)*self.size:ii*self.size,:] - L[(ii-1)*self.size:ii*self.size,:].multiply(cum)
            cum = (cum + tmp[ii-1])
        L = sp.vstack(tmp,format='csr')
        return L

    def get_total_infection_paths(self):
        self.total_infection_paths = np.zeros((1,self.runtime+1),dtype=np.float32)[0]
        #self.reduced_total_infection_paths = np.zeros((1,self.runtime+1),dtype=np.float32)[0]
        C = [sp.csr_matrix((self.size,self.size),dtype=np.float32) for ii in range(self.runtime+1)]
        C[0] = self.zm[0].asfptype()
        self.total_infection_paths[0] = np.float32(C[0].nnz)/self.size**2
        #self.reduced_total_infection_paths[0] = np.float32(np.count_nonzero(C[0].sum(axis=0)))/self.size
        for ii in range(1,self.runtime+1):
            C[ii] = C[ii-1] + self.zm[ii].asfptype()
            self.total_infection_paths[ii] = np.float32(C[ii].nnz)/self.size**2
            #self.reduced_total_infection_paths[ii] = np.float32(np.count_nonzero(C[ii].sum(axis=0)))/self.size
        return

    def get_current_infection_paths(self):
        self.current_infection_paths = np.zeros((1,self.runtime+1),dtype=np.double)[0]
        #self.reduced_current_infection_paths = np.zeros((1,self.runtime+1),dtype=np.double)[0]
        for ii in range(0,self.runtime+1):
            self.current_infection_paths[ii] = np.double(self.zm[ii].nnz)/self.size**2
        return

    def get_infection_time(self):
        self.infection_time = sp.csr_matrix((self.size,self.size),dtype=int)
        self.infection_time_distribution = np.zeros((1,self.runtime))[0]
        self.mean_infection_time = np.zeros((2,self.size))
        self.mean_infected_time = np.zeros((2,self.size))
        self.mean_infection_time_distribution = np.zeros((2,self.size))
        self.mean_infected_time_distribution = np.zeros((2,self.size))
        added = sp.eye(self.size, self.size, dtype=int, format='csr')
        add = sp.csr_matrix((self.size,self.size),dtype=int)
        for ii in range(1,self.runtime+1):
            add = self.zm[ii] - self.zm[ii].multiply(added)
            self.infection_time = self.infection_time + add*ii
            added = added + add
            self.infection_time_distribution[ii-1] = add.nnz
        self.infection_time = np.array(self.infection_time.todense())
        self.infection_time_distribution = np.double(self.infection_time_distribution)/self.size**2
        for ii in range(self.size):
            el = self.infection_time[ii,:][self.infection_time[ii,:]>0]
            if el!=[] :
                self.mean_infected_time[0,ii] = np.mean(el)
                self.mean_infected_time[1,ii] = np.std(el)
            else:
                self.mean_infected_time[0,ii] = 0
                self.mean_infected_time[1,ii] = 0
            
            el = self.infection_time[:,ii][self.infection_time[:,ii]>0]
            if el!=[] :
                self.mean_infection_time[0,ii] = np.mean(el)
                self.mean_infection_time[1,ii] = np.std(el)
            else:
                self.mean_infection_time[0,ii] = 0
                self.mean_infection_time[1,ii] = 0
        
        ind = np.argsort(self.mean_infection_time[0,:])[::-1]
        self.mean_infection_time_distribution[0,:] = self.mean_infection_time[0,ind]
        self.mean_infection_time_distribution[1,:] = self.mean_infection_time[1,ind]
        ind = np.argsort(self.mean_infected_time[0,:])[::-1]
        self.mean_infected_time_distribution[0,:] = self.mean_infected_time[0,ind]
        self.mean_infected_time_distribution[1,:] = self.mean_infected_time[1,ind]
        
#        if np.sum(self.mean_infected_time)==0:
#            print 'keiner wurde infiziert'
#        if np.sum(self.mean_infection_time)==0:
#            print 'keiner hat infiziert' 




        #Tmax = np.amax(self.infection_time)
        #Thist = np.zeros((self.size,Tmax),dtype=int)
        #for ii in range(self.size):
        #    Thist[ii,:], Tbin = np.histogram(self.infection_time[ii,:], bins=range(Tmax+1))
        #self.infection_time_distribution = np.sum(Thist[:,1::],axis=0)

    def generiereZM(self, graph="sexual_contacts.dat", size=100, prob=0.01, runtime=100, isjob=False, mode="SIR", memory=10000):
        if ((str(graph) == "sociopatterns_hypertext.dat") or (str(graph) == "sexual_contacts.dat")):
            A = readdata(str(graph))
            self.size = A[0].shape[0]
            self.prob = 0
            self.runtime = len(A)
            self.graph = str(graph)
            self.isjob = bool(isjob)
            self.mode = str(mode)
            self.memory = int(memory)
            if self.mode == 'SI':
                self.zm = self.SI_load(A)
            elif self.mode == 'SIS':
                self.zm = self.SIS_load(A)
            elif self.mode == 'SIR':
                self.memory = self.memory+1
                self.zm = self.SIR_load(A)
            else:
                print('define desease spreading model: SI, SIS, SIR')        
                
        else:
            self.graph = ''
            self.size = int(size)
            self.prob = float(prob)
            self.runtime = int(runtime)
            self.isjob = bool(isjob)
            self.mode = str(mode)
            self.memory = int(memory)
            if self.mode == 'SI':
                self.zm = self.SI_random()
            elif self.mode == 'SIS':
                self.zm = self.SIS_random()
            elif self.mode == 'SIR':
                self.memory = self.memory+1
                self.zm = self.SIR_random()
            else:
                print('define desease spreading model: SI, SIS, SIR')



#===============================================================================
# Es folgt die Testumgebung:
#===============================================================================
#'load either sociopatterns_hypertext.dat, sexual_contacts.dat or nothing: []')        
if __name__ == "__main__":
    test = Zugangsmatrix()
    test.generiereZM()
    test.save()
#    test.load('/users/stud/koher/master/python/2014_02_26/results/m=3000_SIR.npz')
    #test.get_total_infection_paths()
    #test.get_current_infection_paths()
#    test.get_infection_time()
#    import matplotlib.pyplot as plt
#    Tbin = range(1,len(test.infection_time_distribution)+1)
#    plt.bar(Tbin, test.infection_time_distribution, width=1, color='black',)
#    plt.show()
