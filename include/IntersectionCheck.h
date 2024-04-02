#define CHECKDIM 3

enum QUADRANT{
    RIGHT = 0,
    LEFT = 1,
    MIDDLE = 2
};

template <typename value_type>
bool RayBoxCheck(value_type minB[CHECKDIM], value_type maxB[CHECKDIM], value_type origin[CHECKDIM], value_type dir[CHECKDIM], value_type coord[CHECKDIM])
/* value_type minB[CHECKDIM], maxB[CHECKDIM];        box */
/* value_type origin[CHECKDIM], dir[CHECKDIM];        ray */
/* value_type coord[CHECKDIM];            hit point */
{
    bool inside = true;
    QUADRANT quadrant[CHECKDIM];
    int i;
    int whichPlane;
    value_type maxT[CHECKDIM];
    value_type candidatePlane[CHECKDIM];

    /* Find candidate planes; this loop can be avoided if
       rays cast all from the eye(assume perpsective view) */
    for (i=0; i<CHECKDIM; i++)
    {
        if(origin[i] < minB[i]) {
            quadrant[i] = LEFT;
            candidatePlane[i] = minB[i];
            inside = false;
        }else if (origin[i] > maxB[i]) {
            quadrant[i] = RIGHT;
            candidatePlane[i] = maxB[i];
            inside = false;
        }else    {
            quadrant[i] = MIDDLE;
        }
    }

    /* Ray origin inside bounding box */
    if(inside)    {
        coord = origin;
        return true;
    }

    /* Calculate T distances to candidate planes */
    for (i = 0; i < CHECKDIM; i++)
    {
        if (quadrant[i] != MIDDLE && dir[i] !=0.)
            maxT[i] = (candidatePlane[i]-origin[i]) / dir[i];
        else
            maxT[i] = -1.;
    }

    /* Get largest of the maxT's for final choice of intersection */
    whichPlane = 0;
    for (i = 1; i < CHECKDIM; i++)
    {
        if (maxT[whichPlane] < maxT[i])
            whichPlane = i;
    }

    /* Check final candidate actually inside box */
    if (maxT[whichPlane] < 0.) return false;
    for (i = 0; i < CHECKDIM; i++)
    {
        if (whichPlane != i) {
            coord[i] = origin[i] + maxT[whichPlane] *dir[i];
            if (coord[i] < minB[i] || coord[i] > maxB[i])
                return false;
        } else {
            coord[i] = candidatePlane[i];
        }
    }

    return true;                /* ray hits box */
}

template <typename value_type>
bool RayTriCheck(value_type tri[3][3], value_type origin[3], value_type dir[3])
{
    // calculate t where origin + t * dir is the intersection point of the ray to the triangle plane
    value_type e0[3] = {tri[1][0] - tri[0][0], tri[1][1] - tri[0][1], tri[1][2] - tri[0][2]};
    value_type e1[3] = {tri[2][0] - tri[0][0], tri[2][1] - tri[0][1], tri[2][2] - tri[0][2]};
    value_type n[3]  = {e0[1]*e1[2] - e0[2]*e1[1], e0[2]*e1[0] - e0[0]*e1[2], e0[0]*e1[1] - e0[1]*e1[0]};

    // plane axis to project
    int i1=1, i2=2;
    if(std::fabs(n[1]) > std::max(std::fabs(n[0]),std::fabs(n[2]))) i1 = 0;
    if(std::fabs(n[2]) > std::max(std::fabs(n[0]),std::fabs(n[1]))) i2 = 0;

    value_type d = -tri[0][0]*n[0] - tri[0][1]*n[1] - tri[0][2]*n[2];
    value_type n_dot_dir = dir[0]*n[0]+dir[1]*n[1]+dir[2]*n[2];
    if(std::fabs(n_dot_dir) < 1e-8) return false;
    value_type t =  -(d+origin[0]*n[0]+origin[1]*n[1]+origin[2]*n[2])/n_dot_dir;
    value_type p[3] = {origin[0]+t*dir[0], origin[1]+t*dir[1], origin[2]+t*dir[2]};

    value_type u0,u1,u2;
    value_type v0,v1,v2;
    u0 = p[i1] - tri[0][i1]; v0 = p[i2] - tri[0][i2];
    u1 = e0[i1]; v1 = e0[i2];
    u2 = e1[i1]; v2 = e1[i2];

    bool inter = false;
    value_type alpha, beta;
    if (u1 == 0) {
        beta = u0 / u2;
        if ((beta >= 0.) && (beta <= 1.)) {
            alpha = (v0 - beta*v2) / v1;
            inter = ((alpha >= 0.) && (alpha + beta) <= 1.);
        }
    }
    else {
        beta = (v0*u1 - u0*v1) / (v2*u1 - u2*v1);
        if ((beta >= 0.) && (beta <= 1.)) {
            alpha = (u0 - beta*u2) / u1;
            inter = ((alpha >= 0) && ((alpha + beta) <= 1.));
        }
    }

    return inter;
}

template <typename value_type>
bool RayPolyCheck()
{
    return false;
}
