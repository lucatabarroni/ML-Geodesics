import matplotlib.pyplot as plt
import numpy as np

def gaussian(x,mu,sigma):
    return np.exp(-(x-mu)**2/(2*sigma*sigma))/(np.sqrt(2*np.pi)*sigma)


dmu   = 2
mu1   = 0
mu2   = mu1 + dmu
mu3   = mu2 + dmu
mu4   = mu3 + dmu
mu5   = mu4 + dmu
mu6   = mu5 + dmu

sigma = 0.5*dmu


mu = np.linspace(mu1-5*sigma,mu6+5*sigma,200)
plt.figure(figsize=(6,5))
plt.plot(mu,gaussian(mu,mu1,sigma),color='red')
plt.plot(mu,gaussian(mu,mu2,sigma),color='red')
plt.plot(mu,gaussian(mu,mu3,sigma),color='red')
plt.plot(mu,gaussian(mu,mu4,sigma),color='red')
plt.plot(mu,gaussian(mu,mu5,sigma),color='red')
plt.plot(mu,gaussian(mu,mu6,sigma),color='red')


plt.plot(mu,   gaussian(mu,mu1,sigma)
              +gaussian(mu,mu2,sigma)
              +gaussian(mu,mu3,sigma)
              +gaussian(mu,mu4,sigma)
              +gaussian(mu,mu5,sigma)
              +gaussian(mu,mu6,sigma)

              
              )
plt.show()



