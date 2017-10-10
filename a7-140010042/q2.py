from core import *
from numpy.random import multivariate_normal as nmdist



def q2(Nx,method,hdx):
	xi = np.linspace(-1,1,Nx+1,endpoint = True)
	Err_fun = []
	Err_derv = []	
	fx = lambda	x: -np.sin(np.pi*x)
	dfx = lambda x: -np.pi*np.cos(np.pi*x)
		
	fxi = np.array(list(map(fx,xi)))
	dfxi = np.array(list(map(dfx,xi)))
	zetaf,zetadf = nmdist([0, 0],[[1e-2, 0],[0, 1e-2]],len(fxi)).T
	fxi += zetadf
	dfxi += zetadf

	x = np.linspace(-1,1,1001,endpoint = True)
	exact_fn = np.array(list(map(fx,x)))
	exact_dfn = np.array(list(map(dfx,x)))
	comp_fn = np.zeros_like(x)
	comp_dfn = np.zeros_like(x)

	if method == "cubic_spline":
		fun_eval = sph_eval_fun_csp
		derv_eval = sph_eval_grad_csp
	else:
		fun_eval = sph_eval_fun_gau
		derv_eval = sph_eval_grad_gau
	for i in range(len(x)):
		comp_fn[i] = fun_eval(x[i],fxi,xi,hdx)
		comp_dfn[i] = derv_eval(x[i],fxi,xi,hdx)
	temp = comp_fn[125:875] - exact_fn[125:875]
	temp = temp**2
	temp = ((np.sum(temp))**0.5)/(751)
	Err_fun.append(temp)
	
	temp = comp_dfn[125:875] - exact_dfn[125:875]
	temp = temp**2
	temp = ((np.sum(temp))**0.5)/(751)
	Err_derv.append(temp)
	return Err_fun,Err_derv,x,exact_fn,exact_dfn,xi,fxi,dfxi,comp_fn,comp_dfn

if __name__ == "__main__":
	Err_fun0,Err_derv0,x,exact_fn,exact_dfn,xi0,fxi0,dfxi0,comp_fn0,comp_dfn0 = q2(20,"cubic_spline",1.1)		
	Err_fun1,Err_derv1,x,exact_fn,exact_dfn,xi1,fxi1,dfxi1,comp_fn1,comp_dfn1 = q2(20,"gaussian",1.1)


	plt.figure(1)
	plt.plot(x,exact_fn,label = 'Exact sol')
	plt.plot(x,comp_fn0,label = 'Cubic spline kernel' + str(Err_fun0))
	plt.plot(x,comp_fn1,label = 'Gaussian kernel' + str(Err_fun0))
	plt.plot(xi0,fxi0,'bo',label = 'Sample')
	
	plt.legend(loc='best', shadow=True)
	plt.title('Effect of noise in f(x) approximation')
	plt.savefig('noise.png')

	plt.figure(2)
	plt.plot(x,exact_dfn,label = 'Exact sol')
	plt.plot(x,comp_dfn0,label = 'Cubic spline kernel' + str(Err_fun0))
	plt.plot(x,comp_dfn1,label = 'Gaussian kernel' + str(Err_fun1))
	plt.plot(xi0,dfxi0,'bo',label = 'Sample')
	
	plt.legend(loc='best', shadow=True)
	plt.title('Effect of noise in f(x) approximation')
	plt.savefig('dnoise.png')
