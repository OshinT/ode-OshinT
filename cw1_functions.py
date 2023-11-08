import numpy as np

def forward_euler2(a,b,d,g,t,t_max,Del_t,Hi,Li):
    
    """
    Forward euler method for Lokta-Volterra Model 2 species
    Parameters for the model - a,b,d,g are aplha, beta, delta, gamma respectively.
    t : start time
    t_max: maximum runtime 
    Del_t: constant timestep
    Hi: initial hare population
    Li: initial lynx population
    Returns lists of hare, lynx and time data
    """
    # Create lists to store data for plotting
    hare=[]
    lynx=[]
    time=[]
    # Recursive timestepping loop
    while t < t_max:
        time.append(t)
        H_t1 = Hi + Del_t*(a*Hi - b*Hi*Li)
        L_t1 = Li + Del_t*(d*Hi*Li - g*Li)
        t=t+Del_t
        # append newest values to lists
        hare.append(H_t1)
        lynx.append(L_t1)
        # update Hi and Li to continue timestepping
        Hi = H_t1
        Li = L_t1
    return (hare, lynx, time)


def forward_euler3(a,b,d,g,ep,et,r,t,t_max,Del_t,Hi,Li,Wi):
    
    """
    Forward euler method for Lokta-Volterra Model 3 species
    Parameters for the model - a,b,d,g,ep,et,r are aplha, beta, delta, gamma, epsilon, eta, rho respectively.
    t : start time
    t_max: maximum runtime 
    Del_t: constant timestep
    Hi: initial hare population
    Li: initial lynx population
    Wi: initial wolf population
    Returns lists of hare, lynx, wolf and time data
    """
    # Create lists to store data for plotting
    hare=[]
    lynx=[]
    wolf=[]
    time=[]
    # Recursive timestepping loop
    while t < t_max:
        time.append(t)
        H_t1 = Hi + Del_t*(a*Hi - b*Hi*Li)
        L_t1 = Li + Del_t*(d*Hi*Li - g*Li -ep*Li*Wi)
        W_t1 = Wi + Del_t*(-et*Wi + r*Wi*Li)
        t=t+Del_t
        # append newest values to lists
        hare.append(H_t1)
        lynx.append(L_t1)
        wolf.append(W_t1)
        # update Hi, Li and Wi to continue timestepping
        Hi = H_t1
        Li = L_t1
        Wi = W_t1
    return (hare, lynx, wolf, time)


def modforward_euler2(a,b,d,g,t,t_max,Del_t,Hi,Li):
    
    """
    Modified forward euler method for Lokta-Volterra Model 2 species
    Parameters for the model - a,b,d,g are aplha, beta, delta, gamma respectively.
    t : start time
    t_max: maximum runtime 
    Del_t: constant timestep
    Hi: initial hare population
    Li: initial lynx population
    Returns lists of hare, lynx and time data
    """
    # Create lists to store data for plotting
    hare=[]
    lynx=[]
    time=[]
    # Recursive timestepping loop
    while t < t_max:
        time.append(t)
        H_t1 = Hi + Del_t*(a*Hi - b*Hi*Li)
        L_t1 = Li + Del_t*(d*H_t1*Li - g*Li)
        t=t+Del_t
        # append newest values to lists
        hare.append(H_t1)
        lynx.append(L_t1)
        # update Hi and Li to continue timestepping
        Hi = H_t1
        Li = L_t1
    return (hare, lynx, time)


def modforward_euler3(a,b,d,g,ep,et,r,t,t_max,Del_t,Hi,Li,Wi):
    
    """
    Modified forward euler method for Lokta-Volterra Model 3 species
    Parameters for the model - a,b,d,g,ep,et,r are aplha, beta, delta, gamma, epsilon, eta, rho respectively.
    t : start time
    t_max: maximum runtime 
    Del_t: constant timestep
    Hi: initial hare population
    Li: initial lynx population
    Wi: initial wolf population
    Returns lists of hare, lynx, wolf and time data
    """
    # Create lists to store data for plotting
    hare=[]
    lynx=[]
    wolf=[]
    time=[]
    # Recursive timestepping loop
    while t < t_max:
        time.append(t)
        H_t1 = Hi + Del_t*(a*Hi - b*Hi*Li)
        L_t1 = Li + Del_t*(d*H_t1*Li - g*Li -ep*Li*Wi)
        W_t1 = Wi + Del_t*(-et*Wi + r*Wi*L_t1)
        t=t+Del_t
        # append newest values to lists
        hare.append(H_t1)
        lynx.append(L_t1)
        wolf.append(W_t1)
        # update Hi, Li and Wi to continue timestepping
        Hi = H_t1
        Li = L_t1
        Wi = W_t1
    return (hare, lynx, wolf, time)



def forward_euler(rhs, Xi, dt):
   
    """
    This function takes in a rhs function (lorenz), a numpy array Xi (containing the values of X at time level i) and the 
    timestep dt,and applies the forward euler method.
    Returns a numpy array Xip1, containing the values of X at time level i+1. 
    """
    Xip1_x = Xi[0] + dt*rhs[0]
    Xip1_y = Xi[1] + dt*rhs[1]
    Xip1_z = Xi[2] + dt*rhs[2]
    Xip1 = np.array([Xip1_x,Xip1_y,Xip1_z])
    return (Xip1)

def lorenz_fe(t,t_max,Xi,dt):
   
    """
    This function takes a numpy array Xi (containing the values of X at time level i), the timestep dt and the total run time 
    (t -> t_max) ,and applies the forward euler method to the lorenz value at Xi to get the integrated solution Xip1 
    (X at time level i+1).
    Returns a numpy array X of, containing the values of X at times t -> t_max. Also returns a numpy array of the times.
    """
    #create lists
    X = Xi.copy()
    time = [0]
    
    while t <= t_max:
        rhs = lorenz(Xi)
        Xip1 = forward_euler(rhs, Xi, dt)  # updates Xip1 using the forward_euler function
        X = np.vstack((X, Xip1))  # appends Xip1 to Xi and stores the result as X
        Xi = Xip1  # update Xi
        t += dt    # increment t
        time.append(t)
    return(X,time)

def lorenz_equation(t, X, s=10, r=28, b=8/3):
   
    """
    Function used in SciPy.integrate 
    Function calcualtes the values of dx/dt ,dy/dt and dz/dt for a value of X using the Lorenz equations.
    Returns a numpy array of (dx/dt ,dy/dt, dz/dt)
    """
    # note the optional arguments sigma, r and b that take default values
    x,y,z =X 
    dx=s*(y-x)
    dy=-(x*z) + r*x- y
    dz=x*y- b*z
    return np.array([dx,dy,dz])

def varying_parameters_lorenz(s,r,b):
    """
    function which returns lorenz_equation with varied parameters set by s,r and b
    """
    
    def lorenz_equation(t, X, s=s, r=r, b=b):

        # note the optional arguments sigma, r and b that take default values
        x,y,z =X 
        dx=s*(y-x)
        dy=-(x*z) + r*x- y
        dz=x*y- b*z
        return np.array([dx,dy,dz])
    return lorenz_equation



def RK4(rhs,X, dt):
    
    """
    Applies the Runge-Kutta method of integration for a function 'rhs' of some value X at time t. 
    Returns X_1 which is the value of X at time t+dt
    """    
    k1=rhs(X)
    k2=rhs(X+0.5*dt*k1)
    k3=rhs(X+0.5*dt*k2)
    k4=rhs(X+dt*k3)

    X_1 = X+dt*(k1 + 2*k2 + 2*k3 + k4) / 6.0

    return X_1


def lorenz(X, s=10, r=28, b=8/3):
   
    """
    Applies the Lorenz equations to a value X at time t.
    Returns the Lorenz values of d_X
    """
    d_x = s*(X[1]-X[0])
    d_y = -(X[0]*X[2]) + r*X[0]- X[1]
    d_z = X[0]*X[1]- b*X[2]
    d_X= np.array([d_x,d_y,d_z])
    return(d_X)



def BredVectors(X0,Xp,t,dt,max_t,n):
   
    """
    X0: control state
    Xp: perturbed state
    t: initialising time
    dt: timestep
    max_t: max run time
    n: number of timesteps
    
    Integrates the control and perturbed states using RK4 for intervals of 8 timesteps then evaluates the bredvectors and their 
    growth rates. 
    Returns np arrays of control simulation data (Xc), perturbed simulation data (Xper), bred vectors (bred), growth rates 
    (growth), times each growth rate/ bred vector was calulated (growth_time), checking 8 timesteps were run when evaluating Xc 
    and Xper (timesteps).
    """
    Xn = X0
    Xpn = X0 + Xp
    bv= Xpn-X0

    # creates lists to store data:
    time = [0]
    Xc = X0.copy()  
    Xper = Xpn.copy()
    bred = bv.copy()
    growth = np.array([0])
    growth_time = [0]
    timesteps= []
    
    # Integrate the models and calculating the bred vectors and their growth rates
    while t < max_t:

        #Run RK4 integration for n=8 timesteps 
        i=0
        while i < n*dt:     
            Xn_p1 = RK4(rhs=lorenz, X=Xn, dt=dt) 
            Xpn_p1 = RK4(rhs=lorenz, X=Xpn, dt=dt) 
            #update values
            Xn = Xn_p1 
            Xpn = Xpn_p1
            i += dt
            t += dt
            #store time and timestep data
            time.append(t)
            timesteps.append(i)

        #calculate growth rate by evaluating the bred vector
        Xn_1=Xn
        Xpn_1=Xpn
        b = np.array(Xpn_1) - np.array(Xn_1)
        g= (1/(n*dt))*np.log(np.linalg.norm(b)/np.linalg.norm(Xp)) # np.linalg.norm calculates the magnitude of a vector
        b_r = np.linalg.norm(Xp)*b/np.linalg.norm(b)

        # Storing results for control, perturbed, Bred vectors, growth rates, times growth rates were evaluated.
        Xc = np.vstack((Xc, Xn_1))  
        Xper = np.vstack((Xper, Xpn_1))
        bred = np.vstack((bred, b))
        growth = np.vstack((growth, g))
        growth_time.append(t)
        # update values
        Xn = Xn_1 
        Xpn = Xn_1 + b_r
        
    return(Xc, Xper, bred, growth, growth_time, timesteps)



def lorenz_ENSO(X, sigma=10, r=28, b=8/3,S=1,c=1,o=-11, tao=0.1):
    """
    Applies the coupled Lorenz equations for ENSO to a value X at time t.
    Returns the coupled Lorenz values of d_X
    """
    d_x1 = sigma*(X[1]-X[0]) - c*((S*X[3])+o)
    d_y1 = r*X[0] - X[1] - X[0]*X[2] + c*((S*X[4])+o)
    d_z1 = X[0]*X[1] - b*X[2] + c*X[5]
    
    d_x2 = tao*sigma*(X[4]-X[3]) - c*(X[0]+o) 
    d_y2 = tao*r*X[3] - tao*X[4] - tao*S*X[3]*X[5] + c*(X[1]+o) 
    d_z2 = tao*S*X[3]*X[4]- tao*b*X[5] -c*X[2] 

    return(np.array([d_x1, d_y1, d_z1, d_x2, d_y2, d_z2]))

def fast_and_slow_ENSO(Xn,t,dt,max_t,rhs=lorenz_ENSO):
    """
    Xn: coupled state
    t: initialising time
    dt: timestep
    max_t: max run time
    rhs: right hand side equation
    
    Integrates the coupled states using RK4 method for rhs= lorenz_ENSO 
    Returns np arrays of coupled simulation data (X), times each integration was calulated (time)
    """

    # create a list to store the time and a numpy array to store X:
    time = [0]
    X = Xn.copy()  # create a copy of the initial conditions - without the copy, changing X will change X0

    # Integrate the models
    while t < max_t:

        Xn_p1 = RK4(rhs=rhs,X=Xn, dt=dt) # update Xn_p1 using RK4 function
        X = np.vstack((X, Xn_p1))
        #update values
        Xn = Xn_p1 
        t += dt
        time.append(t)
    return (X,time)







