prog: ray-tracing.c
	gcc ray-tracing.c -lm -o prog
clean:
	rm -f prog image.ppm
