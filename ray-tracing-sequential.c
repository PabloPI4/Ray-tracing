#include "ray-tracing.h"

sph *spheres;
int n_spheres;
cam camera;
lght light;
int width = 1920;
int height = 1080;
double vectorScreenH[3];
double vectorScreenV[3];
short *screen;
double firstPixel[3]; //This is the pixel in the lower left corner

FILE* image;

int main(int argc, char *argv[]) {
    image = fopen("image.ppm","w");
    if(!image) { /*error with the end image*/
	    perror("fopen");
        exit(3);
    }
    readObjects(argc, argv);

    screen = (short **) malloc(sizeof(short *) * width * height * 3);

    ray_global();

    free(screen);
}

void readObjects(int argc, char *argv[]) {
    FILE *readObjects;
    double pos[3];
    double r;
    int end,i;
    char stdbol = 0; 
    if (argc < 2) {
        readObjects = stdin;
        stdbol = 1;
    }
    else if (argc == 2) {
        printf("Objects are going to be readed from the file named: %s\n", argv[1]);
        readObjects = fopen(argv[1], "r");
        if (readObjects == NULL) {
            perror("fprintf");
            fprintf(stderr, "Error opening the file named: %s\n", argv[1]);
        }
    }
    else {
        fprintf(stderr, "Incorrect number of arguments. Passed %d arguments while this program needs just 1\n", argc - 1);
    }
    if(stdbol) printf("Write the camera's position. Format: x y z vector(x) vector(y) vector(z)\n");
    if (fscanf(readObjects, "%lf%lf%lf%lf%lf%lf", camera.center, camera.center + 1, camera.center + 2, camera.direction, camera.direction + 1, camera.direction + 2) < 6) {
        fprintf(stderr, "Error reading camera\n");
        exit(1);
    }
    if(stdbol) printf("Write the light's position. Format: x y z intensity\n");
    if (fscanf(readObjects, "%lf%lf%lf%lf", light.pos, light.pos + 1, light.pos + 2, light.itsty) < 3) {
        fprintf(stderr, "Error reading camera\n");
        exit(2);
    }
    end = 0;
    while(!end) {
        if(stdbol) printf("Reading the center and radius of the spheres. Format: x y z r\n");
        if (fscanf(readObjects, "%lf%lf%lf%lf", pos, pos + 1, pos + 2, &r) < 4) {
            if(stdbol) fprintf(stderr, "End of reading\n");
            end = 1;
        }
        else {
            spheres = (sph*) realloc(spheres, ++n_spheres * sizeof(sph));
            for (i = 0; i < 3; i++) {
                spheres[n_spheres - 1].center[i] = pos[i];
            }
            spheres[n_spheres - 1].r = r;
        }
    }

    if(stdbol) fclose(readObjects);
}

void ray_global() {
    double pointInit[3];
    //CALCULAR LOS VECTORES DE LA PANTALLA

    for (int i = 0; i < width * height * 3; i+=3) {
        ray_tracing(screen + i, i);
    }
}

void ray_tracing(short *posScreen, int x) { //PASAR ESTOS CALCULOS A ray_global PARA PODER HACER ESTA FUNCIÓN RECURSIVA Y REDEFINIR ARGUMENTOS
    double fromPoint[3];
    ln ray;

    for (int i = 0; i < 3; i++) {
        fromPoint[i] = firstPixel[i] + vectorScreenH[i] * (x % width) + vectorScreenV[i] * (x/width);
        ray.startPoint[i] = camera.center[i];
        ray.vector[i] = fromPoint[i] - camera.center[i];
    }

    //SI collisionWithLight, ENTONCES RETURN COLOR DE LUZ CON SU INTENSIDAD
    
    //SI NO, HACER EL CALCULO DEL PUNTO CON EL QUE COLISIONA EN ALGUNA ESFERA

    //SI NO COLISIONA CON NINGUNA ESFERA, ENTONCES RETURN COLOR FONDO CON INTENSIDAD 0

    //SI COLISIONA CON UNA ESFERA, ENTONCES MODIFICAR EL COLOR CON LO OBTENIDO EN ray_tracing(argumentos siguiente esfera) Y RETURN NUEVO COLOR E INTENSIDAD DE LA LUZ
}

sph *calculateCollisions(ln *ray, double point[3]) {
    //This function calculates the sphere that the ray collides with
    sph *sphere;
    double distance = __DBL_MAX__; /*Initialised to DBL_MAX because the first distance 
    calculated should be a candidate for the global distance. With this value we are 
    sure that every distance is lower than this one*/
    double a = scalarProduct(ray->vector, ray->vector);
    double b;
    double c;
    double insideSqrt;
    double distanceSol;
    double vector[3];
    double newFirstPoint[3];

    for (int i = 0; i < n_spheres; i++) {
        /*We calculate the values of a, b and c from the equation that calculates the 
        intersection points between a line and a sphere. More details are given in README.md*/
        b = 2*(ray->vector[0]*(ray->startPoint[0] - spheres[i].center[0]) + ray->vector[1] * 
        (ray->startPoint[1] - spheres[i].center[1]) + ray->vector[2] * (ray->startPoint[2] - spheres[i].center[2]));

        vector[0] = ray->startPoint[0] - spheres[i].center[0];
        vector[1] = ray->startPoint[1] - spheres[i].center[1];
        vector[2] = ray->startPoint[2] - spheres[i].center[2];
        c = scalarProduct(vector, vector);
        c -= (sphere->r * sphere->r);
        insideSqrt = b*b - 4*a*c;

        /*With a, b and c we calculate if the equation has any solution. We know that if 
        b^2 - 4ac < 0 the equation has no solutions*/
        if (sqrt < 0) {
            continue;
        }

        /*Then we calculate the intersection points and evaluate if they are between 
        the point from where the ray starts and infinite in the direction of the ray.
        If this happens, then we evaluate if the distance between this point and the origin 
        of the ray is lower than distances calculated with other spheres collided.
        If this happens, then the sphere collided is equal to the sphere of this iteration*/
        distanceSol = (b + sqrt(insideSqrt))/(2 * a);
        for (int j = 0; j < 2; j++) {
            int behind = 1;

            for (int j = 0; j < 3; j++) {
                newFirstPoint[i] = ray->startPoint[i] + ray->vector[i] * distanceSol;
                vector[i] = newFirstPoint[i] - point[i];

                if ((vector[i] == 0 && ray->vector[i] != 0) || vector[i]/ray->vector[i] < 0) {
                    behind = 0;
                    break;
                }
            }

            if (!behind) {
                continue;
            }

            distanceSol = calculateDistance(point, newFirstPoint);

            if (distanceSol < distance) {
                distance = distanceSol;
                sphere = spheres + i;
            }

            distanceSol = (b - sqrt(insideSqrt))/(2 * a);
        }
    }

    for (int i = 0; i < 3; i++) {
        point[i] = newFirstPoint[i];
    }

    return sphere;
}

int collisionWithLight(ln *ray) {
    if ((light.pos[0] - ray->startPoint[0])/ray->vector[0] == (light.pos[1] - ray->startPoint[1])/ray->vector[1] && 
    (light.pos[0] - ray->startPoint[0])/ray->vector[0] == (light.pos[2] - ray->startPoint[2])/ray->vector[2]) {
        return 1;
    }
    else {
        return 0;
    }
}

ln *calculateReflex(ln *ray, sph *sphere, double point[3]) {
    double vectorStdSph[3];
    double newPoint[3];
    /*For calculating the reflexed ray it is necesary to calculate a second point of the line.
    For calculating this point, we calculate a point of symmetry (a point in the line that goes 
    from the center of the sphere and the intersection point of the ray and the sphere).
    Then we calculate another point of the ray, so we can calculate the symmetric point of this 
    one using the point of symmetry.
    The last step is calling calculateRay passing this 2 points (the calculated and the 
    intersection point)*/
    for (int i = 0; i < 3; i++) {
        vectorStdSph[i] = (point[i] - sphere->center[i])/sphere->r;
        newPoint[i] = 2*(point[i] + vectorStdSph[i]) - (point[i] - ray->vector[i]);
    }
    return calculateRay(point, newPoint, ray);
}

ln *calculateRay(double point1[3], double point2[3], ln *ray) {
    //If ray is not inicializated we do it
    if (!ray) { /*ray == NULL*/
        if ((ray = (ln *) malloc(sizeof(ln))) == NULL) {
            perror("malloc");
            exit(3);
        }
    }
    /*We calculate the vector that goes from point1 to point2 and then we standardizate 
    it dividing it's coordinates by the scalar product of vector*vector*/
    for (int i = 0; i < 3; i++) {
        ray->startPoint[i] = point1[i];
        ray->vector[i] = point2[i] - point1[i];
    }
    double std = sqrt(scalarProduct(ray->vector, ray->vector));
    for (int i = 0; i < 3; i++) {
        ray->vector[i] /= std;
    }
    return ray;
}

double calculateDistance(double point1[3], double point2[3]) {
    /*For calculating the distance between two points first we calculate the
    vector that goes from point1 to point2, and then we calculate the scalarProduct 
    of VectorDistance*VectorDistance*/
    double vectorDistance[3];
    for (int i = 0; i < 3; i++) {
        vectorDistance[i] = point1[i] - point2[i];
    }
    return sqrt(scalarProduct(vectorDistance, vectorDistance));
}

double scalarProduct(double vector1[3], double vector2[3]) {
    return vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2];
}
