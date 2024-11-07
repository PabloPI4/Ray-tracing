#include "ray-tracing.h"

sph *spheres;
int n_spheres;
cam camera;
double light[3];

FILE* image;

int main(int argc, char *argv[]) {
    image = fopen("image.ppm","w");
    if(!image) {
	perror("fopen");
        return 3;
    }
    readObjects(argc, argv);
}

void readObjects(int argc, char *argv[]) {
    FILE *readObjects;
    double pos[3];
    double r;

    if (argc < 2) {
        readObjects = stdin;
    }
    else if (argc == 2) {
        printf("Objects are going to be readed from file named: %s\n", argv[1]);
        readObjects = fopen(argv[1], "r");
        if (readObjects == NULL) {
            fprintf(stderr, "Error opening file named: %s\n", argv[1]);
        }
    }
    else {
        fprintf(stderr, "Number of arguments incorrect. Passed %d arguments and this program only need 1\n", argc - 1);
    }

    printf("Write the position of the camera: x y z vector(x) vector(y) vector(z)\n");
    if (fscanf(readObjects, "%lf%lf%lf%lf%lf%lf", camera.center, camera.center + 1, camera.center + 2, camera.direction, camera.direction + 1, camera.direction + 2) < 6) {
        fprintf(stderr, "Error reading camera\n");
        exit(1);
    }

    printf("Write the position of the light: x y z\n");
    if (fscanf(readObjects, "%lf%lf%lf", light, light + 1, light + 2) < 3) {
        fprintf(stderr, "Error reading camera\n");
        exit(2);
    }

    int end = 0;
    while(!end) {
        printf("Reading the center and radius of the spheres: x y z r\n");
        if (fscanf(readObjects, "%lf%lf%lf%lf", pos, pos + 1, pos + 2, &r) < 4) {
            fprintf(stderr, "End of reading\n");
            end = 1;
        }
        else {
            spheres = (sph *) realloc(spheres, ++n_spheres * sizeof(struct sphere));
            for (int i = 0; i < 3; i++) {
                spheres[n_spheres - 1].center[i] = pos[i];
            }
            spheres[n_spheres - 1].r = r;
        }
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
    if (ray == NULL) {
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

    double std = scalarProduct(ray->vector, ray->vector);
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

    return scalarProduct(vectorDistance, vectorDistance);
}

double scalarProduct(double vector1[3], double vector2[3]) {
    return sqrt(vector1[1]*vector2[1] + vector1[2]*vector2[2] + vector1[3]*vector2[3]);
}
