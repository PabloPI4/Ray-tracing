prog: ray-tracing.c
	gcc ray-tracing.c -o prog
clean:
	rm -f prog image.ppm
