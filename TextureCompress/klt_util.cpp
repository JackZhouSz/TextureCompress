/*********************************************************************
 * klt_util.c
 *********************************************************************/

 /* Standard includes */
#include <assert.h>
#include <stdlib.h>  /* malloc() */
#include <math.h>		/* fabs() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "pnmio.h"
#include "klt.h"
#include "klt_util.h"

#ifndef max
#define max(a,b)	((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)	((a) < (b) ? (a) : (b))
#endif

#define max3(a,b,c)	((a) > (b) ? max((a),(c)) : max((b),(c)))

#define min3(a,b,c)	((a) < (b) ? min((a),(c)) : min((b),(c)))

/*********************************************************************/

float _KLTComputeSmoothSigma(
	KLT_TrackingContext tc)
{
	return (tc->smooth_sigma_fact * max(tc->window_width, tc->window_height));
}

/*********************************************************************
 * _KLTCreateFloatImage
 */

_KLT_FloatImage _KLTCreateFloatImage(int ncols, int nrows)
{
	_KLT_FloatImage floatImg;
	int nbytes = sizeof(_KLT_FloatImageRec) + ncols * nrows * sizeof(float);
	floatImg = (_KLT_FloatImage)malloc(nbytes);
	if (floatImg == NULL)
		KLTError("(_KLTCreateFloatImage)  Out of memory");
	floatImg->ncols = ncols;
	floatImg->nrows = nrows;
	floatImg->data = (float*)(floatImg + 1);

	return (floatImg);
}

/*********************************************************************
 * _KLTFreeFloatImage
 */

void _KLTFreeFloatImage(_KLT_FloatImage floatImg)
{
	free(floatImg);
}

/*********************************************************************
 * _KLTPrintSubFloatImage
 */

void _KLTPrintSubFloatImage(_KLT_FloatImage floatimg, int x0, int y0, int width, int height)
{
	int ncols = floatimg->ncols;
	int offset;
	int i, j;

	assert(x0 >= 0);
	assert(y0 >= 0);
	assert(x0 + width <= ncols);
	assert(y0 + height <= floatimg->nrows);

	fprintf(stderr, "\n");
	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			offset = (j + y0) * ncols + (i + x0);
			fprintf(stderr, "%6.2f ", *(floatimg->data + offset));
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");

}

/*********************************************************************
 * _KLTWriteFloatImageToPGM
 */

void _KLTWriteFloatImageToPGM(_KLT_FloatImage img, const char* filename)
{
	int npixs = img->ncols * img->nrows;
	float mmax = -999999.9f, mmin = 999999.9f;
	float fact;
	float* ptr;
	uchar* byteImg, * ptrOut;
	int i;

	/* Calculate minimum and maximum values of float image */
	ptr = img->data;
	for (int i = 0; i < npixs; i++) {
		mmax = max(mmax, *ptr);
		mmin = min(mmin, *ptr);
		ptr++;
	}

	/* Allocate memory to hold converted image */
	byteImg = (uchar*)malloc(npixs * sizeof(uchar));

	/* Convert image from float to uchar */
	fact = 255.0f / (mmax - mmin);
	ptr = img->data;
	ptrOut = byteImg;
	for (int i = 0; i < npixs; i++) {
		*ptrOut++ = (uchar)((*ptr++ - mmin) * fact);
	}

	/* Write uchar image to PGM */
	pgmWriteFile(filename, byteImg, img->ncols, img->nrows);

	/* Free memory */
	free(byteImg);
}

/* for affine mapping */
void _KLTWriteAbsFloatImageToPGM(_KLT_FloatImage img, char* filename, float scale)
{
	int npixs = img->ncols * img->nrows;
	float fact;
	float* ptr;
	uchar* byteImg, * ptrout;
	int i;
	float tmp;

	/* Allocate memory to hold converted image */
	byteImg = (uchar*)malloc(npixs * sizeof(uchar));

	/* Convert image from float to uchar */
	fact = 255.0f / scale;
	ptr = img->data;
	ptrout = byteImg;
	for (i = 0; i < npixs; i++) {
		tmp = (float)(fabs(*ptr++) * fact);
		if (tmp > 255.0) tmp = 255.0;
		*ptrout++ = (uchar)tmp;
	}

	/* Write uchar image to PGM */
	pgmWriteFile(filename, byteImg, img->ncols, img->nrows);

	/* Free memory */
	free(byteImg);
}
