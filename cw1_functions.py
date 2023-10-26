
def forward_euler2(a,b,d,g,t,t_max,Del_t,Hi,Li):
    """
    Forward euler method for Lokta-Volterra Model 2 species
    Parameters for the model - a,b,d,g are aplha, beta, delta, gamma respectively.
    t : start time
    t_max: maximum runtime 
    Del_t: constant timestep
    Hi: initial hare population
    Li: initial lynx population
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
    # This function takes in a rhs function (lorenz), a numpy array Xi (containing the values of X at time level i) and the timestep dt,
    # and returns a numpy array Xip1, containing the values of X at time level i+1
    Xip1_x = Xi[0] + dt*rhs[0]
    Xip1_y = Xi[1] + dt*rhs[1]
    Xip1_z = Xi[2] + dt*rhs[2]
    Xip1 = np.array([Xip1_x,Xip1_y,Xip1_z])
    return (Xip1)

def lorenz_fe(t,t_max,Xi,dt):
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
    # note the optional arguments sigma, r and b that take default values
    x,y,z =X 
    dx=s*(y-x)
    dy=-(x*z) + r*x- y
    dz=x*y- b*z
    return np.array([dx,dy,dz])


def RK4(X, dt):
    
    k1=lorenz(X)
    k2=lorenz(X+0.5*dt*k1)
    k3=lorenz(X+0.5*dt*k2)
    k4=lorenz(X+dt*k3)

    X_1 = X+dt*(k1 + 2*k2 + 2*k3 + k4) / 6.0

    return X_1


def lorenz(X, s=10, r=28, b=8/3):
    d_x = s*(X[1]-X[0])
    d_y = -(X[0]*X[2]) + r*X[0]- X[1]
    d_z = X[0]*X[1]- b*X[2]
    d_X=np.array([d_x,d_y,d_z])
    return(d_X)



def BredVectors(X0,Xp,t,dt,max_t,n):
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
            Xn_p1 = RK4(X=Xn, dt=dt) 
            Xpn_p1 = RK4(X=Xpn, dt=dt) 
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



