
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

