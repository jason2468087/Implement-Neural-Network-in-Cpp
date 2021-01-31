#pragma once

class AMV
{
public:
	static int sum(int*, int);
	static float sum(float*, int);
	static double sum(double*, int);

	static float avg(int*, int);
	static float avg(float*, int);
	static double avg(double*, int);

	static int maxIdx(int*, int);
	static int maxIdx(float*, int);
	static int maxIdx(double*, int);

	static int max(int*, int);
	static float max(float*, int);
	static double max(double*, int);

	static int minIdx(int*, int);
	static int minIdx(float*, int);
	static int minIdx(double*, int);

	static int min(int*, int);
	static float min(float*, int);
	static double min(double*, int);

	static void addV(int*, int*, int*, int);
	static void addV(float*, float*, float*, int);
	static void addV(double*, double*, double*, int);

	static void subV(int*, int*, int*, int);
	static void subV(float*, float*, float*, int);
	static void subV(double*, double*, double*, int);

	static void mulV(int*, int*, int*, int);
	static void mulV(float*, float*, float*, int);
	static void mulV(double*, double*, double*, int);

	static void divV(float*, float*, float*, int);
	static void divV(double*, double*, double*, int);

	static void addC(int*, int*, int, int);
	static void addC(float*, float*, float, int);
	static void addC(double*, double*, double, int);

	static void subC(int*, int*, int, int);
	static void subC(float*, float*, float, int);
	static void subC(double*, double*, double, int);

	static void mulC(int*, int*, int, int);
	static void mulC(float*, float*, float, int);
	static void mulC(double*, double*, double, int);

	static void divC(float*, float*, float, int);
	static void divC(double*, double*, double, int);

	static float dot(float*,float*,int);
	static double dot(double*, double*, int);

	static float dotV(float**, float*,int, int);
	static double dotV(double**, double*,int, int);
};
