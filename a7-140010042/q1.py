from core import *


def q1(method,hdx):
	Nxi = [10, 20, 30, 50, 80, 100, 30]
	Err_fun = []
	Err_derv = []

	for Nx in Nxi:
		xi = np.linspace(-1,1,Nx+1,endpoint = True)
		
		fx = lambda	x: -np.sin(np.pi*x)
		dfx = lambda x: -np.pi*np.cos(np.pi*x)
		
		fxi = np.array(list(map(fx,xi)))
		dfxi = np.array(list(map(dfx,xi)))
		
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

	return Nxi, Err_fun, Err_derv, x, exact_fn, comp_fn, exact_dfn, comp_dfn
	

if __name__ == "__main__":
	hdx = [1.1, 1.2, 1.8, 2.4]
	######## Cubic spline kernel
	Nx0,Err_fun0,Err_derv0,x,exact_fn,comp_fn0,exact_dfn,comp_dfn0 = q1("cubic_spline",hdx[0])
	Nx1,Err_fun1,Err_derv1,x,exact_fn,comp_fn1,exact_dfn,comp_dfn1 = q1("cubic_spline",hdx[1])
	Nx2,Err_fun2,Err_derv2,x,exact_fn,comp_fn2,exact_dfn,comp_dfn2 = q1("cubic_spline",hdx[2])
	Nx3,Err_fun3,Err_derv3,x,exact_fn,comp_fn3,exact_dfn,comp_dfn3 = q1("cubic_spline",hdx[3])
	csp_fn = comp_fn1;csp_dfn = comp_dfn1
	plt.figure(1)
	plt.loglog(Nx0[0:-1],Err_fun0[0:-1],label = 'hdx = ' + str(hdx[0]))
	plt.loglog(Nx1[0:-1],Err_fun1[0:-1],label = 'hdx = ' + str(hdx[1]))
	plt.loglog(Nx2[0:-1],Err_fun2[0:-1],label = 'hdx = ' + str(hdx[2]))
	plt.loglog(Nx3[0:-1],Err_fun3[0:-1],label = 'hdx = ' + str(hdx[3]))
	plt.legend(loc='best', shadow=True)
	plt.title('Error plot for f(x) for various sample size and different hdx values')
	plt.savefig('fun_err_csp.png')


	plt.figure(2)
	plt.loglog(Nx0[0:-1],Err_derv0[0:-1],label = 'hdx = ' + str(hdx[0]))
	plt.loglog(Nx1[0:-1],Err_derv1[0:-1],label = 'hdx = ' + str(hdx[1]))
	plt.loglog(Nx2[0:-1],Err_derv2[0:-1],label = 'hdx = ' + str(hdx[2]))
	plt.loglog(Nx3[0:-1],Err_derv3[0:-1],label = 'hdx = ' + str(hdx[3]))
	plt.legend(loc='best', shadow=True)
	plt.title('Error plot for df(x)/dx for various sample size and different hdx values')
	plt.savefig('derv_err_csp.png')

	plt.figure(3)
	plt.plot(x,exact_fn,label = 'exact_csp')
	plt.plot(x,comp_fn0,label = 'hdx = ' + str(hdx[0]))
	plt.plot(x,comp_fn1,label = 'hdx = ' + str(hdx[1]))
	plt.plot(x,comp_fn2,label = 'hdx = ' + str(hdx[2]))
	plt.plot(x,comp_fn3,label = 'hdx = ' + str(hdx[3]))
	plt.legend(loc='best', shadow=True)
	plt.title('Function aprroximations')
	plt.savefig('fun_csp.png')

	plt.figure(4)
	plt.plot(x,exact_dfn,label = 'exact_csp')
	plt.plot(x,comp_dfn0,label = 'hdx = ' + str(hdx[0]))
	plt.plot(x,comp_dfn1,label = 'hdx = ' + str(hdx[1]))
	plt.plot(x,comp_dfn2,label = 'hdx = ' + str(hdx[2]))
	plt.plot(x,comp_dfn3,label = 'hdx = ' + str(hdx[3]))
	plt.legend(loc='best', shadow=True)
	plt.title('Derivative aprroximations')
	plt.savefig('derv_csp.png')

	plt.figure(5)
	plt.plot(x[250:300],exact_dfn[250:300],label = 'exact_csp')
	plt.plot(x[250:300],comp_dfn0[250:300],label = 'hdx = ' + str(hdx[0]))
	plt.plot(x[250:300],comp_dfn1[250:300],label = 'hdx = ' + str(hdx[1]))
	plt.plot(x[250:300],comp_dfn2[250:300],label = 'hdx = ' + str(hdx[2]))
	plt.plot(x[250:300],comp_dfn3[250:300],label = 'hdx = ' + str(hdx[3]))
	plt.legend(loc='best', shadow=True)
	plt.title('Derivative aprroximations close up')
	plt.savefig('derv_csp_clup.png')



	######## Gaussian kernel
	Nx0,Err_fun0,Err_derv0,x,exact_fn,comp_fn0,exact_dfn,comp_dfn0 = q1("gaussian",hdx[0])
	Nx1,Err_fun1,Err_derv1,x,exact_fn,comp_fn1,exact_dfn,comp_dfn1 = q1("gaussian",hdx[1])
	Nx2,Err_fun2,Err_derv2,x,exact_fn,comp_fn2,exact_dfn,comp_dfn2 = q1("gaussian",hdx[2])
	Nx3,Err_fun3,Err_derv3,x,exact_fn,comp_fn3,exact_dfn,comp_dfn3 = q1("gaussian",hdx[3])
	gau_fn = comp_fn1;gau_dfn = comp_dfn1
	
	plt.figure(6)
	plt.loglog(Nx0[0:-1],Err_fun0[0:-1],label = 'hdx = ' + str(hdx[0]))
	plt.loglog(Nx1[0:-1],Err_fun1[0:-1],label = 'hdx = ' + str(hdx[1]))
	plt.loglog(Nx2[0:-1],Err_fun2[0:-1],label = 'hdx = ' + str(hdx[2]))
	plt.loglog(Nx3[0:-1],Err_fun3[0:-1],label = 'hdx = ' + str(hdx[3]))
	plt.legend(loc='best', shadow=True)
	plt.title('Error plot for f(x) for various sample size and different hdx values')
	plt.savefig('fun_err_gau.png')


	plt.figure(7)
	plt.loglog(Nx0[0:-1],Err_derv0[0:-1],label = 'hdx = ' + str(hdx[0]))
	plt.loglog(Nx1[0:-1],Err_derv1[0:-1],label = 'hdx = ' + str(hdx[1]))
	plt.loglog(Nx2[0:-1],Err_derv2[0:-1],label = 'hdx = ' + str(hdx[2]))
	plt.loglog(Nx3[0:-1],Err_derv3[0:-1],label = 'hdx = ' + str(hdx[3]))
	plt.legend(loc='best', shadow=True)
	plt.title('Error plot for df(x)/dx for various sample size and different hdx values')
	plt.savefig('derv_err_gau.png')

	plt.figure(8)
	plt.plot(x,exact_fn,label = 'exact_csp')
	plt.plot(x,comp_fn0,label = 'hdx = ' + str(hdx[0]))
	plt.plot(x,comp_fn1,label = 'hdx = ' + str(hdx[1]))
	plt.plot(x,comp_fn2,label = 'hdx = ' + str(hdx[2]))
	plt.plot(x,comp_fn3,label = 'hdx = ' + str(hdx[3]))
	plt.legend(loc='best', shadow=True)
	plt.title('Function aprroximations')
	plt.savefig('fun_gau.png')

	plt.figure(9)
	plt.plot(x,exact_dfn,label = 'exact_csp')
	plt.plot(x,comp_dfn0,label = 'hdx = ' + str(hdx[0]))
	plt.plot(x,comp_dfn1,label = 'hdx = ' + str(hdx[1]))
	plt.plot(x,comp_dfn2,label = 'hdx = ' + str(hdx[2]))
	plt.plot(x,comp_dfn3,label = 'hdx = ' + str(hdx[3]))
	plt.legend(loc='best', shadow=True)
	plt.title('Derivative aprroximations')
	plt.savefig('derv_gau.png')

	plt.figure(10)
	plt.plot(x[250:300],exact_dfn[250:300],label = 'exact_csp')
	plt.plot(x[250:300],comp_dfn0[250:300],label = 'hdx = ' + str(hdx[0]))
	plt.plot(x[250:300],comp_dfn1[250:300],label = 'hdx = ' + str(hdx[1]))
	plt.plot(x[250:300],comp_dfn2[250:300],label = 'hdx = ' + str(hdx[2]))
	plt.plot(x[250:300],comp_dfn3[250:300],label = 'hdx = ' + str(hdx[3]))
	plt.legend(loc='best', shadow=True)
	plt.title('Derivative aprroximations close up')
	plt.savefig('derv_gau_clup.png')

	plt.figure(11)
	plt.plot(x,exact_fn,label = 'Exact sol')
	plt.plot(x,csp_fn,label = 'Cubic spline kernel')
	plt.plot(x,gau_fn,label = 'Gaussian kernel')
	plt.legend(loc='best', shadow=True)
	plt.title('Kernel comparison for f(x) at hdx = ' + str(hdx[1]))
	plt.savefig('comp_fn.png')

	plt.figure(12)
	plt.plot(x,exact_dfn,label = 'Exact sol')
	plt.plot(x,csp_dfn,label = 'Cubic spline kernel')
	plt.plot(x,gau_dfn,label = 'Gaussian kernel')
	plt.legend(loc='best', shadow=True)
	plt.title('Kernel comparison for df(x)/dx at hdx = ' + str(hdx[1]))
	plt.savefig('comp_dfn.png')
