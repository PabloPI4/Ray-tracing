#include "ray-casting.h"

sph *spheres;
pln *walls;
int n_walls;
int n_spheres;
cam camera;
lght light;
int width;
int height;
double vectorScreenH[3];
double vectorScreenV[3];
unsigned char *screen;
double firstPixel[3]; //This is the pixel in the lower left corner

FILE* image;

int main(int argc, char *argv[]) {
    image = fopen("cast.ppm","w");
    if(!image) { /*error with the end image*/
	    perror("fopen");
        exit(3);
    }
    readObjects(argc, argv);

    screen = (unsigned char *) malloc(width * height * 3);

    ray_global();

    writeImage();

    free(screen);
}

void readObjects(int argc, char *argv[]) {
    FILE *readObjects;
    double pos[3];
    double color[3];
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
        exit(10);
    }

    if(stdbol) printf("Write the size of the screen: FHD (1920x1080), QHD (2560x1440) or UHD (3840x2160)\n");
    char res[4];
    if (fgets(res, 4, readObjects) == NULL) {
        fprintf(stderr, "incorrect resolution\n");
        exit(5);
    }

    if (strcmp(res, "FHD") == 0) {
        width = 1920;
        height = 1080;
    }
    else if (strcmp(res, "QHD") == 0) {
        width = 2560;
        height = 1440;
    }
    else if (strcmp(res, "UHD") == 0) {
        width = 3840;
        height = 2160;
    }
    else if (strcmp(res, "VLI") == 0) {
        width = 13200;
        height = 7425;
    }
    else {
        fprintf(stderr, "incorrect resolution\n");
        exit(5);
    }


    if(stdbol) printf("Write the camera's position. Format: x y z vector(x) vector(z)\n");
    if (fscanf(readObjects, "%lf%lf%lf%lf%lf", camera.center, camera.center + 1, camera.center + 2, camera.direction, camera.direction + 2) < 5) {
        fprintf(stderr, "Error reading camera\n");
        exit(1);
    }
    camera.direction[1] = 0;
    double cameraScalar = scalarProduct(camera.direction, camera.direction);
    for (int i = 0; i < 3; i++) {
        camera.direction[i] /= cameraScalar;
    }

    if(stdbol) printf("Write the light's position. Format: x y z intensity\n");
    if (fscanf(readObjects, "%lf%lf%lf%lf", light.center, light.center + 1, light.center + 2, &light.itsty) < 4) {
        fprintf(stderr, "Error reading camera\n");
        exit(2);
    }
    light.r = 20;

    walls = (pln *) malloc(sizeof(pln) * 6);

    walls[0].color[0] = 255;
    walls[0].color[1] = 255;
    walls[0].color[2] = 255;
    n_walls = 6;

    if(stdbol) printf("Write the ceiling color. Format: colorR colorG colorB\n");
    if (fscanf(readObjects, "%lf%lf%lf", color, color + 1, color + 2) < 3) {
        fprintf(stderr, "Error reading ceiling color\n");
        exit(2);
    }
    walls[1].color[0] = color[0];
    walls[1].color[1] = color[1];
    walls[1].color[2] = color[2];
    if(stdbol) printf("Write the left wall color. Format: colorR colorG colorB\n");
    if (fscanf(readObjects, "%lf%lf%lf", color, color + 1, color + 2) < 3) {
        fprintf(stderr, "Error reading left wall color\n");
        exit(2);
    }
    walls[2].color[0] = color[0];
    walls[2].color[1] = color[1];
    walls[2].color[2] = color[2];
    if(stdbol) printf("Write the right wall color. Format: colorR colorG colorB\n");
    if (fscanf(readObjects, "%lf%lf%lf", color, color + 1, color + 2) < 3) {
        fprintf(stderr, "Error reading right wall color\n");
        exit(2);
    }
    walls[3].color[0] = color[0];
    walls[3].color[1] = color[1];
    walls[3].color[2] = color[2];
    if(stdbol) printf("Write the front wall color. Format: colorR colorG colorB\n");
    if (fscanf(readObjects, "%lf%lf%lf", color, color + 1, color + 2) < 3) {
        fprintf(stderr, "Error reading front wall color\n");
        exit(2);
    }
    walls[4].color[0] = color[0];
    walls[4].color[1] = color[1];
    walls[4].color[2] = color[2];
    if(stdbol) printf("Write the back wall color. Format: colorR colorG colorB\n");
    if (fscanf(readObjects, "%lf%lf%lf", color, color + 1, color + 2) < 3) {
        fprintf(stderr, "Error reading back wall color\n");
        exit(2);
    }
    walls[5].color[0] = color[0];
    walls[5].color[1] = color[1];
    walls[5].color[2] = color[2];

    end = 0;
    while(!end) {
        if(stdbol) printf("Reading the center, radius and color of the spheres. Format: x y z r colorR colorG colorB\n");
        if (fscanf(readObjects, "%lf%lf%lf%lf%lf%lf%lf", pos, pos + 1, pos + 2, &r, color + 0, color + 1, color + 2) < 7) {
            if(stdbol) fprintf(stderr, "End of reading\n");
            end = 1;
        }
        else {
            spheres = (sph*) realloc(spheres, ++n_spheres * sizeof(sph));
            for (i = 0; i < 3; i++) {
                spheres[n_spheres - 1].center[i] = pos[i];
                spheres[n_spheres - 1].color[i] = color[i];
                spheres[n_spheres - 1].reflection[i] = color[i]/255.0;

            }
            spheres[n_spheres - 1].r = r;
        }
    }

    double centerPoint[3];
    centerPoint[0] = camera.center[0] + camera.direction[0];
    centerPoint[1] = camera.center[1] + camera.direction[1];
    centerPoint[2] = camera.center[2] + camera.direction[2];

    vectorScreenV[0] = 0;
    vectorScreenV[1] = 25.6/width;
    vectorScreenV[2] = 0;

    //HACER PRODUCTO VECTORIAL
    vectorScreenH[0] = camera.direction[1] * vectorScreenV[2] - camera.direction[2] * vectorScreenV[1];
    vectorScreenH[1] = camera.direction[2] * vectorScreenV[0] - camera.direction[0] * vectorScreenV[2];
    vectorScreenH[2] = camera.direction[0] * vectorScreenV[1] - camera.direction[1] * vectorScreenV[0];

    firstPixel[0] = centerPoint[0] - vectorScreenH[0]*(width/2) - vectorScreenV[0]*(height/2);
    firstPixel[1] = centerPoint[1] - vectorScreenH[1]*(width/2) - vectorScreenV[1]*(height/2);
    firstPixel[2] = centerPoint[2] - vectorScreenH[2]*(width/2) - vectorScreenV[2]*(height/2);

    walls[0].a = -vectorScreenV[0];
    walls[0].b = -vectorScreenV[1];
    walls[0].c = -vectorScreenV[2];
    walls[0].d = -(walls[0].a + walls[0].b * (centerPoint[1] + vectorScreenV[1] * (height/2)));

    walls[1].a = vectorScreenV[0];
    walls[1].b = vectorScreenV[1];
    walls[1].c = vectorScreenV[2];
    walls[1].d = -(walls[1].a + walls[1].b * (centerPoint[1] - vectorScreenV[1] * (height/2)));

    walls[2].a = -vectorScreenH[0];
    walls[2].b = -vectorScreenH[1];
    walls[2].c = -vectorScreenH[2];
    walls[2].d = -(walls[2].a + walls[2].c * (centerPoint[2] + vectorScreenH[2] * (width/2)));

    walls[3].a = vectorScreenH[0];
    walls[3].b = vectorScreenH[1];
    walls[3].c = vectorScreenH[2];
    walls[3].d = -(walls[3].a + walls[3].c * (centerPoint[2] - vectorScreenH[2] * (width/2)));

    walls[4].a = -camera.direction[0];
    walls[4].b = -camera.direction[1];
    walls[4].c = -camera.direction[2];
    walls[4].d = -(walls[4].a * (centerPoint[0] + camera.direction[0] * 10));

    walls[5].a = camera.direction[0];
    walls[5].b = camera.direction[1];
    walls[5].c = camera.direction[2];
    walls[5].d = -(walls[5].a * (centerPoint[0] - camera.direction[0] * 7));

    if(stdbol) fclose(readObjects);
}

void ray_global() {
    double pointInit[3];
    double x[3];
    cllsn collision;
    cllsn cols[20];
    int colPos;
    ln ray;
    int noLight;
    sph *sphereCollided;
    pln *wallCollided;
    //CALCULAR LOS VECTORES DE LA PANTALLA

    for (int i = 0; i < width * height * 3; i+=3) {
        sphereCollided = NULL;
        wallCollided = NULL;
        colPos = 0;
        noLight = 0;

        for (int j = 0; j < 3; j++) {
            pointInit[j] = firstPixel[j] + vectorScreenH[j] * ((i % (width * 3))/3) + vectorScreenV[j] * (i/(width * 3));
            ray.startPoint[j] = pointInit[j];
            ray.vector[j] = pointInit[j] - camera.center[j];
            x[j] = pointInit[j];
        }

        calculateCollisions(&ray, pointInit, &sphereCollided, &wallCollided, &collision);

        if (colPos > 19 || (sphereCollided == NULL && wallCollided == NULL)) {
            noLight = 1;
        }

        if (noLight) {
            screen[i] = 0;
            screen[i + 1] = 0;
            screen[i + 2] = 0;
        }
        else if (wallCollided == NULL) {
            screen[i] = sphereCollided->color[0];
            screen[i + 1] = sphereCollided->color[1];
            screen[i + 2] = sphereCollided->color[2];
        }
        else {
            screen[i] = wallCollided->color[0];
            screen[i + 1] = wallCollided->color[1];
            screen[i + 2] = wallCollided->color[2];
        }
    }
}

void calculateCollisions(ln *ray, double point[3], sph **originSph, pln **originWall, cllsn *collision) { //REVISAR ESTA FUNCIÓN
    //This function calculates the sphere that the ray collides with
    sph *sphere = NULL;
    pln *plane = NULL;
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
    double returnPoint[3];

    for (int i = 0; i < n_spheres; i++) {
        if (*originSph != NULL && *originSph == spheres + i) { //If the sphere is the same as origin
            continue;
        }
        /*We calculate the values of a, b and c from the equation that calculates the 
        intersection points between a line and a sphere. More details are given in README.md*/

        vector[0] = ray->startPoint[0] - spheres[i].center[0];
        vector[1] = ray->startPoint[1] - spheres[i].center[1];
        vector[2] = ray->startPoint[2] - spheres[i].center[2];
        b = -2*(ray->vector[0]*vector[0] + ray->vector[1]*vector[1] + ray->vector[2]*vector[2]);
        c = scalarProduct(vector, vector);
        c -= (spheres[i].r * spheres[i].r);
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
            int behind = 0;

            for (int k = 0; k < 3; k++) {
                newFirstPoint[k] = ray->startPoint[k] + ray->vector[k] * distanceSol;
                vector[k] = newFirstPoint[k] - point[k];

                if (vector[k]/ray->vector[k] < 0) {
                    behind = 1;
                    break;
                }
            }

            if (behind) {
                continue;
            }

            distanceSol = calculateDistance(point, newFirstPoint);

            if (distanceSol < distance) {
                distance = distanceSol;
                sphere = spheres + i;
                returnPoint[0] = newFirstPoint[0];
                returnPoint[1] = newFirstPoint[1];
                returnPoint[2] = newFirstPoint[2];
            }

            distanceSol = (b - sqrt(insideSqrt))/(2 * a);
        }
    }

    for (int i = 0; i < n_walls; i++) {
        int behind = 0;

        if (*originWall != NULL && *originWall == walls + i) {
            continue;
        }

        insideSqrt = ray->vector[0] * walls[i].a + ray->vector[1] * walls[i].b + ray->vector[2] * walls[i].c;

        if (insideSqrt == 0) {
            continue;
        }

        distanceSol = -(ray->startPoint[0] * walls[i].a + ray->startPoint[1] * walls[i].b + ray->startPoint[2] * walls[i].c + walls[i].d) / insideSqrt;

        for (int j = 0; j < 3; j++) {
            newFirstPoint[j] = ray->startPoint[j] + ray->vector[j] * distanceSol;
            vector[j] = newFirstPoint[j] - point[j];

            if (vector[j]/ray->vector[j] < 0) {
                behind = 1;
                break;
            }
        }

        if (behind) {
            continue;
        }

        distanceSol = calculateDistance(point, newFirstPoint);

        if (distanceSol < distance) {
            distance = distanceSol;
            sphere = NULL;
            plane = walls + i;
            returnPoint[0] = newFirstPoint[0];
            returnPoint[1] = newFirstPoint[1];
            returnPoint[2] = newFirstPoint[2];
        }
    }

    if (sphere != NULL) {
        collision->distance = calculateDistance(point, returnPoint);
        collision->angle = calculateAngle(ray, returnPoint, sphere->center);
        collision->sphere = sphere;
        collision->plane = NULL;

        for (int i = 0; i < 3; i++) {
            point[i] = returnPoint[i];
        }
    }
    else if (plane != NULL) {
        vector[0] = plane->a;
        vector[1] = plane->b;
        vector[2] = plane->c;
        collision->distance = calculateDistance(point, returnPoint);
        collision->angle = calculateAngle(ray, returnPoint, vector);
        collision->plane = plane;
        collision->sphere = NULL;

        for (int i = 0; i < 3; i++) {
            point[i] = returnPoint[i];
        }
    }

    *originSph = sphere;
    *originWall = plane;
}

int collisionWithLight(ln *ray) {
    double a = scalarProduct(ray->vector, ray->vector);
    double b;
    double c;
    double vector[3];
    double newFirstPoint[3];

    vector[0] = ray->startPoint[0] - light.center[0];
    vector[1] = ray->startPoint[1] - light.center[1];
    vector[2] = ray->startPoint[2] - light.center[2];
    b = -2*(ray->vector[0]*vector[0] + ray->vector[1]*vector[1] + ray->vector[2]*vector[2]);
    c = scalarProduct(vector, vector);
    c -= (light.r * light.r);
    double insideSqrt = b*b - 4*a*c;

    if (insideSqrt < 0) {
        return 0;
    }
    else {
        double distanceSol = (b + sqrt(insideSqrt))/(2 * a);
        int behind = 0;
        for (int k = 0; k < 3; k++) {
            newFirstPoint[k] = ray->startPoint[k] + ray->vector[k] * distanceSol;
            vector[k] = newFirstPoint[k] - ray->startPoint[k];

            if (vector[k]/ray->vector[k] < 0) {
                behind = 1;
                break;
            }
        }

        if (behind) {
            return 0;
        }
        else {
            return 1;
        }
    }
}

void calculateRay(double point1[3], double point2[3], ln *ray) {
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

double calculateAngle(ln *ray, double point[3], double centerSphere[3]) {
    double vectorSphere[3];
    for (int i = 0; i < 3; i++) {
        vectorSphere[i] = point[i] - centerSphere[i];
    }

    return acos((ray->vector[0]*vectorSphere[0] + ray->vector[1]*vectorSphere[1] + ray->vector[2]*vectorSphere[2]) / 
    (sqrt(scalarProduct(ray->vector, ray->vector)) * sqrt(scalarProduct(vectorSphere, vectorSphere))));
}

void writeImage() {
    fprintf(image, "P6 %d %d 255\n", width, height);
    fwrite(screen, 1, width*height*3, image);
}