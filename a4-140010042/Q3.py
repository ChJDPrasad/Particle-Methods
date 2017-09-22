from VPM import *
import integrator as itg

def Q3(linear = True):
	#plt.figure()
	if linear == True:
		string = ' of Linear Gamma Dist'
		s = 'q2lin_vort_traj.png'	
	else:
		string = ' of Const Gamma Dist'	
		s = 'q2const_vort_traj.png'
	
	vort_strg = 2.0*np.pi
	vort_pos = 1.5j	
	cy_rad = 1.0  										#	cylinder of radius 
	V_inf = 0.0 + 0.0j	
	npanel = 100
	thetas = np.linspace(0.0,2.0*np.pi,num=npanel,endpoint = False)
	vert = lambda th: -cy_rad*np.cos(th) + 1j*cy_rad*np.sin(th)
	vert = list(map(vert,thetas))
	#print(vert)

	#rad = np.linspace(1.5,5,num=1,endpoint = True)
	rad = [1.5]		
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
		exact = lambda t: 1j*rad[j]*np.cos(omeg*t) + rad[j]*np.sin(omeg*t)
		position_t, t = itg.simulate(sld_bodies,V_inf,vort_elems,3,0.05)
		exact = list(map(exact,t))			
		error[j] = np.sum(abs(position_t[:,0] - exact[:]))/len(t)
		
		plt.figure()
		plt.plot(position_t[:,0].real,position_t[:,0].imag,label = 'vortex')
		plt.plot(np.array(vert)[:].real,np.array(vert)[:].imag,label = 'cylinder')
		plt.title('Vortex trajectory with 100 panels' + string)
		plt.legend(loc = 'best', prop = {'size' : 10})	
		plt.axis('equal')
		plt.savefig(s)
		plt.figure()
		plt.plot(np.array(exact)[:].real,np.array(exact)[:].imag,label = 'exact')			
		plt.plot(np.array(vert)[:].real,np.array(vert)[:].imag,label = 'cylinder')
		plt.title('Exact trajectory of vortex')
		plt.legend(loc = 'best', prop = {'size' : 10})
		plt.axis('equal')	
		plt.savefig('Exact_sol.png')
					
		

if __name__ == "__main__":
	Q3(linear = False)
	Q3(linear = True)
