#cython: cdivision = True
from libc.math cimport exp, sqrt, pow, erf, M_PI

cpdef double NormFac(double a,int x,int y,int z):
    """Calculate normalization factor for primitive Gaussian-type
       orbital
    
        Keyword Arguments:
        a -- orbital exponential coefficient
        x, y, z --- Angular "quantum" numbers for orbital
        
        return:
        ----------------
        Double - normalization factor
    """
    
    cdef double Norm,Fact

    Fact = factorial2(2 * x - 1) * factorial2(2 * y - 1) * factorial2(2 * z - 1)
    
    Norm = (2 * a / M_PI) ** (.75) * (4 * a) ** (0.5 * (x + y + z)) / sqrt(Fact)
    
    return Norm

cdef int factorial2(int n)nogil:
    ''' Calculate n!!
    
        Keyword Arguments:
        n - integer
        
        Return:
        res - integer, n!!
    '''
    cdef int res = n
    cdef int i
    if n <= 1:
        return 1
    else:
        for i in xrange(n-2,1,-2):
            res *= i
    return res


cdef double Boys0(double T)nogil:
    """Calculate Boys(T,0), equal to 1/2*sqrt(pi)*erf(sqrt(T)/sqrt(T)
        
        Keyword Arguments:
        T -- Value at which Boys0 is evaluated at.
        
        Return:
        Boys0 - float
    """
    cdef double sqT = sqrt(T)
    if sqT<1E-5: return 1
    if sqT>10: return 0.886226925452758014 / sqT
    else: return 0.886226925452758014  * erf(sqT) / sqT
    

cdef int binom(int n, int k)nogil:
    """ Calculates nCk or the kth binomial coefficent for nth power
        binomial
    
        Keyword Arguments:
        n - integer, order of binomial
        k - integer, desired coefficient
        
        Return:
        Res - integer, nCk
    """
    
    cdef int res = 1
    cdef int i
    
    # Since C(n, k) = C(n, n-k)
    if k > n - k :
        k = n - k
 
    # Calculate value of [n * (n-1) *---* (n-k+1)] / [k * (k-1) *---* 1]
    for i in xrange(k):
    
        res *= (n - i)
        res /= (i + 1)
 
    return res

cdef double norm(double x,double y,double z)nogil:
    '''Compute Cartesian norm of three coordinates
    '''
    return x * x + y * y + z * z
