
def forward_euler(a,b,d,g,t,t_max,Del_t,Hi,Li):
    """
    Forward euler method for Lokta-Volterra Model
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


