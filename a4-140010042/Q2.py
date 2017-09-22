from VPM import *
import integrator as itg

def Q2(linear = True):
	plt.figure()
	for nop in np.linspace(20,100,num=3,endpoint = True):
		vort_strg = 2.0*np.pi
		vort_pos = 1.5j	
		cy_rad = 1.0  										#	cylinder of radius 
		V_inf = 0.0 + 0.0j	
		npanel = nop
		thetas = np.linspace(0.0,2.0*np.pi,num=nop,endpoint = False)
		vert = lambda th: -cy_rad*np.cos(th) + 1j*cy_rad*np.sin(th)
		vert = list(map(vert,thetas))
		#print(vert)


		rad = np.linspace(1.5,5,num=20,endpoint = True)
		error = np.zeros_like(rad)
		for j in range(len(rad)):
			#print(j)
			vort_elems = []
			vort_elems.append(vortex(1j*rad[j],vort_strg,0.0))	
			sld_bodies = []
			#print(len(vort_elems))
			sld_bodies.append(sld_bd(vert,linear))
			vel = methodofimages(2*np.pi,1j*rad[j],cy_rad)
			omeg = abs(vel)/rad[j]
			exact = lambda x: 1j*rad[j]*np.cos(omeg*t) + rad[j]*np.sin(omeg*t)
			position_t, t = itg.simulate(sld_bodies,V_inf,vort_elems,3,0.05)
			exact = list(map(exact,t))			
			error[j] = np.sum(abs(position_t[:,0] - exact[:])/abs(exact[-1]))/len(t)
			
		
		plt.plot(rad,error,label = str(nop))

	if linear == True:
		string = ' for Linear Gamma Dist'
		s = 'linear_error.png'
	else:
		string = ' for Const Gamma Dist'
		s = 'const_error.png'	
	plt.title('Error vs radius curve for different no of panels' + string)    		
	plt.legend(loc = 'best', prop = {'size' : 10})		
	plt.savefig('q2' + s)	

if __name__ == "__main__":
	Q2(linear = False)
	Q2(linear = True)
