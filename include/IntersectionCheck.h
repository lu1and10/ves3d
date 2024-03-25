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

    /* Ray origin inside bounding box */
    if(inside)    {
        coord = origin;
        return (true);
    }


    /* Calculate T distances to candidate planes */
    for (i = 0; i < CHECKDIM; i++)
        if (quadrant[i] != MIDDLE && dir[i] !=0.)
            maxT[i] = (candidatePlane[i]-origin[i]) / dir[i];
        else
            maxT[i] = -1.;

    /* Get largest of the maxT's for final choice of intersection */
    whichPlane = 0;
    for (i = 1; i < CHECKDIM; i++)
        if (maxT[whichPlane] < maxT[i])
            whichPlane = i;

    /* Check final candidate actually inside box */
    if (maxT[whichPlane] < 0.) return (false);
    for (i = 0; i < CHECKDIM; i++)
        if (whichPlane != i) {
            coord[i] = origin[i] + maxT[whichPlane] *dir[i];
            if (coord[i] < minB[i] || coord[i] > maxB[i])
                return (false);
        } else {
            coord[i] = candidatePlane[i];
        }
    return (true);                /* ray hits box */
}

template <typename value_type>
bool RayTriCheck(value_type tri[3][3], value_type origin[3], value_type dir[3])
{
    return false;
}

template <typename value_type>
bool RayPolyCheck()
{
    return false;
}

typedef double Point[3];
typedef double Vector[3];
struct Ray {
    Point O, D;
    Point P;
    Vector normal;
} ray;
struct Polygon {
    int n;
    bool interpolate;
};

bool intersect(struct Polygon poly, struct Ray ray, double t, int i1, int i2) {
    double alpha, beta, gamma, u0, u1, u2, v0, v1, v2;
    bool inter;
    int i;
    Point V[5];
    Point P;
    Vector N[3];

    /* the value of t is computed.
     * i1 and i2 come from the polygon description.
     * V is the vertex table for the polygon and N the
     * associated normal vectors.
     */
    P[0] = ray.O[0] + ray.D[0] * t;
    P[1] = ray.O[1] + ray.D[1] * t;
    P[2] = ray.O[2] + ray.D[2] * t;
    u0 = P[i1] - V[0][i1]; v0 = P[i2] - V[0][i2];
    inter = false; i = 2;
    do {
        /* The polygon is viewed as (n-2) triangles. */
        u1 = V[i - 1][i1] - V[0][i1]; v1 = V[i - 1][i2] - V[0][i2];
        u2 = V[i][i1] - V[0][i1]; v2 = V[i][i2] - V[0][i2];

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
    } while ((!inter) && (++i < poly.n));

    if (inter) {
        /* Storing the intersection point. */
        ray.P[0] = P[0]; ray.P[1] = P[1]; ray.P[2] = P[2];
        /* the normal vector can be interpolated now or later. */
        if (poly.interpolate) {
            gamma = 1 - (alpha + beta);
            ray.normal[0] = gamma * N[0][0] + alpha * N[i - 1][0] +
                beta * N[i][0];
            ray.normal[1] = gamma * N[0][1] + alpha * N[i - 1][1] +
                beta * N[i][1];
            ray.normal[2] = gamma * N[0][2] + alpha * N[i - 1][2] +
                beta * N[i][2];
        }
    }
    return inter;
}
