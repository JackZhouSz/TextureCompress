/*********************************************************************
 * pnmio.h
 *********************************************************************/

#ifndef _PNMIO_H_
#define _PNMIO_H_

#include <stdio.h>
#include "base.h"

/**********
* With pgmReadFile and pgmRead, setting img to NULL causes memory
* to be allocated
*/

/**********
* used for reading from/writing to files
*/
uchar* pgmReadFile(const char* fname, uchar* img, int* ncols, int* nrows);

void pgmWriteFile(const char* fname, uchar* img, int ncols, int nrows);

void ppmWriteFileRGB(const char* fname, uchar* redimg, uchar* greenimg, uchar* blueimg, int ncols, int nrows);

/**********
 * used for communicating with stdin and stdout
 */
uchar* pgmRead(FILE* fp, uchar* img, int* ncols, int* nrows);

void pgmWrite(FILE* fp, uchar* img, int ncols, int nrows);

void ppmWrite(FILE* fp, uchar* redimg, uchar* greenimg, uchar* blueimg, int ncols, int nrows);

#endif
