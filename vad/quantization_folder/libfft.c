#define power_opt	1		// 0 for short time power, 1 for filterbank power
#define PI 3.14159265358979323846
#define fft_order 1024
#define fft_frame_size 400
#define bandnum 10

#include <stdio.h>
#include "libfft.h"
#include <math.h>
#include "filterCoeff.h"

extern const float filterCoeff[bandnum][fft_order/2+1];

int bak_array[1024*2];
float input[fft_order*2];
float t_logX[bandnum];
void address_reverse(int fft_array[], char fft_type)
{
//	int bak_array[1024 * 2];
	int num[10];
	int seq[1024];
	int n, i, k;

	for (i = 0; i < 1024; i++)
	{
		k = i;
		for (n = 0; n < 10; n++) { num[n] = (k & 1); k >>= 1; }
		k = num[1] * 512 + num[0] * 256 + num[3] * 128 + num[2] * 64 + num[5] * 32 + num[4] * 16 + num[7] * 8 + num[6] * 4 + num[9] * 2 + num[8] * 1;
		seq[i] = k;
	}

	if ((fft_type == 0) || (fft_type == 4)) for (i = 0; i < 256; i++) bak_array[i] = fft_array[seq[i] >> 2];
	if ((fft_type == 2) || (fft_type == 6)) for (i = 0; i < 1024; i++) bak_array[i] = fft_array[seq[i]];

	if (fft_type == 1) for (i = 0; i < 256; i++) {
		bak_array[seq[i] >> 2] = fft_array[2 * i];
		bak_array[(seq[i] >> 2) + 256] = fft_array[2 * i + 1];
	}
	if (fft_type == 3) for (i = 0; i < 1024; i++) {
		bak_array[seq[i]] = fft_array[2 * i];
		bak_array[seq[i] + 1024] = fft_array[2 * i + 1];
	}

	if (fft_type == 5) for (i = 0; i < 256; i++) {
		bak_array[2 * i] = fft_array[seq[i] >> 2];
		bak_array[2 * i + 1] = fft_array[(seq[i] >> 2) + 256];
	}
	if (fft_type == 7) for (i = 0; i < 1024; i++) {
		bak_array[2 * i] = fft_array[seq[i]];
		bak_array[2 * i + 1] = fft_array[seq[i] + 1024];
	}

	if ((fft_type == 0) || (fft_type == 4)) for (i = 0; i < 256; i++) fft_array[i] = bak_array[i];
	if ((fft_type == 1) || (fft_type == 5)) for (i = 0; i < 512; i++) fft_array[i] = bak_array[i];
	if ((fft_type == 2) || (fft_type == 6)) for (i = 0; i < 1024; i++) fft_array[i] = bak_array[i];
	if ((fft_type == 3) || (fft_type == 7)) for (i = 0; i < 2048; i++) fft_array[i] = bak_array[i];
}

int constant_mul(int a_value, char type) //0:0.707 1:0.60725 2:1.414
{
	int result;
	if (type == 0)      result = (a_value >> 1) + (a_value >> 3) + (a_value >> 4) + (a_value >> 6) + (a_value >> 8);
	else if (type == 1) result = (a_value >> 1) + (a_value >> 3) - (a_value >> 6) - (a_value >> 9) - 1;
	else if (type == 2) result = a_value + (a_value >> 2) + (a_value >> 3) + (a_value >> 5) + (a_value >> 7);
	return(result);
}

void cordic_func(int r_in, int i_in, int *r_out, int *i_out, int K, int CC, int angle[], char type) // 'f':fft 'i':ifft
{
	int k_value, k_flag, tmp_value, i;

	k_value = K;
	*r_out = r_in;
	*i_out = i_in;

	if (k_value >= 512) {
		k_value = k_value - 512;
		k_flag = 2;
	}
	else if (k_value >= 256) {
		k_value = 512 - k_value;
		k_flag = 1;
		tmp_value = *r_out;
		*r_out = *i_out;
		*i_out = tmp_value;
	}
	else {
		k_flag = 0;
	}

	*r_out = constant_mul(*r_out, 1);
	*i_out = constant_mul(*i_out, 1);

	k_value = k_value * pow(2, 7);

	for (i = 0; i < CC; i++)
	{
		tmp_value = *r_out;

		if (k_value >= 0)
		{
			k_value = k_value - angle[i];
			if (type == 'i') {
				*r_out = *r_out - (*i_out >> i);
				*i_out = *i_out + (tmp_value >> i) - 1;
			}
			else {
				*r_out = *r_out + (*i_out >> i);
				*i_out = *i_out - (tmp_value >> i) - 1;
			}
		}
		else {
			k_value = k_value + angle[i];
			if (type == 'i') {
				*r_out = *r_out + (*i_out >> i) - 1;
				*i_out = *i_out - (tmp_value >> i);
			}
			else {
				*r_out = *r_out - (*i_out >> i) - 1;
				*i_out = *i_out + (tmp_value >> i);
			}
		}
	}

	if (k_flag == 2)
	{
		if (type == 'i') {
			tmp_value = *i_out;
			*i_out = *r_out;
			*r_out = -tmp_value - 1;
		}
		else {
			tmp_value = *r_out;
			*r_out = *i_out;
			*i_out = -tmp_value - 1;
		}
	}
	else if (k_flag == 1)
	{
		if (type == 'i') *r_out = (*r_out * -1) - 1;
		else            *i_out = (*i_out * -1) - 1;
	}
}

void hanning_window(int fft_array[], char fft_type, char window_en, int CC, int angle[])
{
	int r_value0, i_value0, r_value1, i_value1, i;

	if (window_en == 1) {
		if (fft_type == 0) for (i = 0; i < 128; i++) {
			cordic_func(fft_array[i], 0, &r_value0, &i_value0, i * 8, CC, angle, 'f');
			cordic_func(fft_array[255 - i], 0, &r_value1, &i_value1, i * 8, CC, angle, 'f');
			fft_array[i] = (fft_array[i] - r_value0) >> 1;
			fft_array[255 - i] = (fft_array[255 - i] - r_value1) >> 1;
		}
		if (fft_type == 1) for (i = 0; i < 256; i++) {
			cordic_func(fft_array[i], 0, &r_value0, &i_value0, i * 4, CC, angle, 'f');
			cordic_func(fft_array[511 - i], 0, &r_value1, &i_value1, i * 4, CC, angle, 'f');
			fft_array[i] = (fft_array[i] - r_value0) >> 1;
			fft_array[511 - i] = (fft_array[511 - i] - r_value1) >> 1;
		}
		if (fft_type == 2) for (i = 0; i < 512; i++) {
			cordic_func(fft_array[i], 0, &r_value0, &i_value0, i * 2, CC, angle, 'f');
			cordic_func(fft_array[1023 - i], 0, &r_value1, &i_value1, i * 2, CC, angle, 'f');
			fft_array[i] = (fft_array[i] - r_value0) >> 1;
			fft_array[1023 - i] = (fft_array[1023 - i] - r_value1) >> 1;
		}
		if (fft_type == 3) for (i = 0; i < 1024; i++) {
			cordic_func(fft_array[i], 0, &r_value0, &i_value0, i * 1, CC, angle, 'f');
			cordic_func(fft_array[2047 - i], 0, &r_value1, &i_value1, i * 1, CC, angle, 'f');
			fft_array[i] = (fft_array[i] - r_value0) >> 1;
			fft_array[2047 - i] = (fft_array[2047 - i] - r_value1) >> 1;
		}
		if (fft_type == 4) for (i = 0; i < 128; i++) {
			cordic_func(fft_array[i], 0, &r_value0, &i_value0, i * 8, CC, angle, 'f');
			cordic_func(fft_array[255 - i], 0, &r_value1, &i_value1, i * 8, CC, angle, 'f');
			fft_array[i] = (fft_array[i] - r_value0) >> 1;
			fft_array[255 - i] = (fft_array[255 - i] - r_value1) >> 1;
		}
	}
}


void radix_2_fft_0(int fft_array[], int a_r_addr, int b_r_addr)
{
	int a_r_value, b_r_value;
	int a_r, b_r;

	a_r_value = fft_array[a_r_addr];
	b_r_value = fft_array[b_r_addr];

	a_r = a_r_value + b_r_value;
	b_r = a_r_value - b_r_value - 1;

	fft_array[a_r_addr] = a_r;
	fft_array[b_r_addr] = b_r;
}

void radix_2_fft_1(int fft_array[], int a_r_addr, int b_r_addr)
{
	int a_r_value, b_r_value;
	int a_r, b_r;

	a_r_value = fft_array[a_r_addr];
	b_r_value = fft_array[b_r_addr];

	a_r = a_r_value;
	b_r = -b_r_value - 1;

	fft_array[a_r_addr] = a_r;
	fft_array[b_r_addr] = b_r;
}

void radix_2_fft_2(int fft_array[], int a_r_addr, int a_i_addr, int b_r_addr, int b_i_addr, int k, int CC, int angle[])
{
	int a_r_value, a_i_value, b_r_value, b_i_value;
	int a_r, a_i, b_r, b_i;

	a_r_value = fft_array[a_r_addr];
	a_i_value = fft_array[a_i_addr];

	cordic_func(fft_array[b_r_addr], fft_array[b_i_addr], &b_r_value, &b_i_value, k * 1, CC, angle, 'f');

	a_r = a_r_value + b_r_value;
	a_i = a_i_value + b_i_value;

	b_r = a_r_value - b_r_value - 1;
	b_i = a_i_value - b_i_value - 1;

	fft_array[a_r_addr] = a_r;
	fft_array[a_i_addr] = b_r;
	fft_array[b_r_addr] = b_i;
	fft_array[b_i_addr] = a_i;
}

void radix_4_fft_0(int fft_array[], int a_r_addr, int b_r_addr, int c_r_addr, int d_r_addr)
{
	int a_r_value, b_r_value, c_r_value, d_r_value;
	int a_r, b_r, c_r, b_i;

	a_r_value = fft_array[a_r_addr];
	b_r_value = fft_array[b_r_addr];
	c_r_value = fft_array[c_r_addr];
	d_r_value = fft_array[d_r_addr];

	a_r = a_r_value + b_r_value + c_r_value + d_r_value;
	c_r = a_r_value - b_r_value + c_r_value - d_r_value - 1;
	b_r = a_r_value - c_r_value - 1;
	b_i = d_r_value - b_r_value - 1;

	fft_array[a_r_addr] = a_r;
	fft_array[b_r_addr] = b_r;
	fft_array[c_r_addr] = c_r;
	fft_array[d_r_addr] = b_i;

}

void radix_4_fft_1(int fft_array[], int a_r_addr, int b_r_addr, int c_r_addr, int d_r_addr)
{
	int a_r_value, b_r_value, c_r_value, d_r_value;
	int a_r, a_i, b_r, b_i;

	a_r_value = fft_array[a_r_addr];
	b_r_value = fft_array[b_r_addr];
	c_r_value = fft_array[c_r_addr];
	d_r_value = fft_array[d_r_addr];

	b_r_value = constant_mul(b_r_value, 0);
	d_r_value = constant_mul(d_r_value, 0);

	a_r = a_r_value + b_r_value - d_r_value - 1;
	a_i = -b_r_value - c_r_value - d_r_value - 2;

	b_r = a_r_value - b_r_value + d_r_value - 1;
	b_i = -b_r_value + c_r_value - d_r_value - 1;

	fft_array[a_r_addr] = a_r;
	fft_array[b_r_addr] = b_r;
	fft_array[c_r_addr] = b_i;
	fft_array[d_r_addr] = a_i;
}

void radix_4_fft_2(int fft_array[], int a_r_addr, int a_i_addr, int b_r_addr, int b_i_addr, int c_r_addr, int c_i_addr, int d_r_addr, int d_i_addr, int k, int CC, int angle[])
{
	int a_r_value, b_r_value, c_r_value, d_r_value;
	int a_i_value, b_i_value, c_i_value, d_i_value;
	int a_r, a_i, b_r, b_i;
	int c_r, c_i, d_r, d_i;

	a_r_value = fft_array[a_r_addr];
	a_i_value = fft_array[a_i_addr];

	cordic_func(fft_array[b_r_addr], fft_array[b_i_addr], &b_r_value, &b_i_value, k * 1, CC, angle, 'f');
	cordic_func(fft_array[c_r_addr], fft_array[c_i_addr], &c_r_value, &c_i_value, k * 2, CC, angle, 'f');
	cordic_func(fft_array[d_r_addr], fft_array[d_i_addr], &d_r_value, &d_i_value, k * 3, CC, angle, 'f');

	a_r = a_r_value + b_r_value + c_r_value + d_r_value;
	a_i = a_i_value + b_i_value + c_i_value + d_i_value;

	b_r = a_r_value + b_i_value - c_r_value - d_i_value - 1;
	b_i = a_i_value - b_r_value - c_i_value + d_r_value - 2;

	c_r = a_r_value - b_r_value + c_r_value - d_r_value - 2;
	c_i = a_i_value - b_i_value + c_i_value - d_i_value - 2;

	d_r = a_r_value - b_i_value - c_r_value + d_i_value - 1;
	d_i = a_i_value + b_r_value - c_i_value - d_r_value - 2;

	fft_array[a_r_addr] = a_r;
	fft_array[a_i_addr] = d_r;
	fft_array[b_r_addr] = b_r;
	fft_array[b_i_addr] = c_r;
	fft_array[c_r_addr] = -c_i - 1;
	fft_array[c_i_addr] = b_i;
	fft_array[d_r_addr] = -d_i - 1;
	fft_array[d_i_addr] = a_i;
}

void radix_2_ifft_0(int fft_array[], int a_r_addr, int b_r_addr)
{
	int a_r_value, b_r_value;
	int a_r, b_r;

	a_r_value = fft_array[a_r_addr];
	b_r_value = fft_array[b_r_addr];

	a_r = a_r_value + b_r_value;
	b_r = a_r_value - b_r_value - 1;

	fft_array[a_r_addr] = a_r >> 1;
	fft_array[b_r_addr] = b_r >> 1;
}

void radix_2_ifft_1(int fft_array[], int a_r_addr, int a_i_addr)
{
	int a_r_value, a_i_value;
	int a_r, b_r;

	a_r_value = fft_array[a_r_addr];
	a_i_value = fft_array[a_i_addr];

	a_r = a_r_value;
	b_r = -a_i_value - 1;

	fft_array[a_r_addr] = a_r >> 0;
	fft_array[a_i_addr] = b_r >> 0;
}

void radix_2_ifft_2(int fft_array[], int a_r_addr, int b_r_addr, int b_i_addr, int a_i_addr, int K, int CC, int angle[])
{
	int a_r_value, b_r_value, a_i_value, b_i_value;
	int a_r, a_i, b_r, b_i;

	a_r_value = fft_array[a_r_addr];
	a_i_value = fft_array[a_i_addr];
	b_r_value = fft_array[b_r_addr];
	b_i_value = fft_array[b_i_addr];

	a_r = a_r_value + b_r_value;
	a_i = a_i_value + b_i_value;

	b_r = a_r_value - b_r_value - 1;
	b_i = a_i_value - b_i_value - 1;

	cordic_func(b_r, b_i, &b_r_value, &b_i_value, K * 1, CC, angle, 'i');

	fft_array[a_r_addr] = a_r >> 1;
	fft_array[b_r_addr] = a_i >> 1;
	fft_array[b_i_addr] = b_r_value >> 1;
	fft_array[a_i_addr] = b_i_value >> 1;
}

void radix_4_ifft_0(int fft_array[], int a_r_addr, int b_r_addr, int c_r_addr, int b_i_addr)
{
	int a_r_value, b_r_value, c_r_value, b_i_value, d_r_value, d_i_value;
	int a_r, b_r, c_r, d_r;

	a_r_value = fft_array[a_r_addr];
	c_r_value = fft_array[c_r_addr];
	d_r_value = fft_array[b_r_addr];
	d_i_value = fft_array[b_i_addr];
	b_r_value = fft_array[b_r_addr];
	b_i_value = -fft_array[b_i_addr] - 1;

	a_r = a_r_value + b_r_value + c_r_value + d_r_value;
	b_r = a_r_value + b_i_value - c_r_value - d_i_value - 1;
	c_r = a_r_value - b_r_value + c_r_value - d_r_value - 1;
	d_r = a_r_value - b_i_value - c_r_value + d_i_value - 2;

	fft_array[a_r_addr] = a_r >> 2;
	fft_array[b_r_addr] = b_r >> 2;
	fft_array[c_r_addr] = c_r >> 2;
	fft_array[b_i_addr] = d_r >> 2;
}


void radix_4_ifft_1(int fft_array[], int a_r_addr, int b_r_addr, int b_i_addr, int a_i_addr)
{
	int a_r_value, a_i_value, b_r_value, b_i_value, c_r_value, c_i_value, d_r_value, d_i_value;
	int a_r, b_r, c_i, d_r;

	a_r_value = fft_array[a_r_addr];
	a_i_value = fft_array[a_i_addr];
	c_r_value = fft_array[b_r_addr];
	c_i_value = -fft_array[b_i_addr] - 1;
	d_r_value = fft_array[b_r_addr];
	d_i_value = fft_array[b_i_addr];
	b_r_value = fft_array[a_r_addr];
	b_i_value = -fft_array[a_i_addr] - 1;

	a_r = a_r_value + b_r_value + c_r_value + d_r_value;
	b_r = a_r_value + b_i_value - c_r_value - d_i_value - 1;
	c_i = a_i_value - b_i_value + c_i_value - d_i_value - 1;
	d_r = a_r_value - b_i_value - c_r_value + d_i_value - 2;

	b_r = constant_mul(b_r, 2);
	d_r = constant_mul(d_r, 2);

	fft_array[a_r_addr] = a_r >> 2;
	fft_array[b_r_addr] = b_r >> 2;
	fft_array[b_i_addr] = -c_i >> 2;
	fft_array[a_i_addr] = -d_r >> 2;
}

void radix_4_ifft_2(int fft_array[], int a_r_addr, int b_r_addr, int c_r_addr, int d_r_addr, int d_i_addr, int c_i_addr, int b_i_addr, int a_i_addr, int K, int CC, int angle[])
{
	int a_r_value, b_r_value, c_r_value, d_r_value;
	int a_i_value, b_i_value, c_i_value, d_i_value;
	int a_r, a_i, b_r, b_i;
	int c_r, c_i, d_r, d_i;

	a_r_value = fft_array[a_r_addr];
	a_i_value = fft_array[a_i_addr];
	d_r_value = fft_array[c_r_addr];
	d_i_value = fft_array[c_i_addr];
	c_r_value = fft_array[d_r_addr];
	c_i_value = -fft_array[d_i_addr] - 1;
	b_r_value = fft_array[b_r_addr];
	b_i_value = -fft_array[b_i_addr] - 1;

	a_r = a_r_value + b_r_value + c_r_value + d_r_value;
	a_i = a_i_value + b_i_value + c_i_value + d_i_value;

	b_r = a_r_value + b_i_value - c_r_value - d_i_value - 1;
	b_i = a_i_value - b_r_value - c_i_value + d_r_value - 2;

	c_r = a_r_value - b_r_value + c_r_value - d_r_value - 1;
	c_i = a_i_value - b_i_value + c_i_value - d_i_value - 1;

	d_r = a_r_value - b_i_value - c_r_value + d_i_value - 2;
	d_i = a_i_value + b_r_value - c_i_value - d_r_value - 1;

	cordic_func(b_r, b_i, &b_r_value, &b_i_value, K * 1, CC, angle, 'i');
	cordic_func(c_r, c_i, &c_r_value, &c_i_value, K * 2, CC, angle, 'i');
	cordic_func(d_r, d_i, &d_r_value, &d_i_value, K * 3, CC, angle, 'i');

	fft_array[a_r_addr] = a_r >> 2;
	fft_array[b_r_addr] = a_i >> 2;
	fft_array[c_r_addr] = b_r_value >> 2;
	fft_array[d_r_addr] = b_i_value >> 2;
	fft_array[d_i_addr] = c_r_value >> 2;
	fft_array[c_i_addr] = c_i_value >> 2;
	fft_array[b_i_addr] = d_r_value >> 2;
	fft_array[a_i_addr] = d_i_value >> 2;
}

void fft_func_256(int fft_array[], int CC, int angle[])
{
	int i, j;

	for (i = 0; i < 64; i++) radix_4_fft_0(fft_array, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3);
	for (i = 0; i < 16; i++) radix_4_fft_0(fft_array, i * 16 + 0 * 4, i * 16 + 1 * 4, i * 16 + 2 * 4, i * 16 + 3 * 4);
	for (i = 0; i < 4; i++) radix_4_fft_0(fft_array, i * 64 + 0 * 16, i * 64 + 1 * 16, i * 64 + 2 * 16, i * 64 + 3 * 16);
	for (i = 0; i < 1; i++) radix_4_fft_0(fft_array, i * 256 + 0 * 64, i * 256 + 1 * 64, i * 256 + 2 * 64, i * 256 + 3 * 64);

	for (i = 0; i < 16; i++) radix_4_fft_1(fft_array, i * 16 + 0 * 4 + 2, i * 16 + 1 * 4 + 2, i * 16 + 2 * 4 + 2, i * 16 + 3 * 4 + 2);
	for (i = 0; i < 4; i++) radix_4_fft_1(fft_array, i * 64 + 0 * 16 + 8, i * 64 + 1 * 16 + 8, i * 64 + 2 * 16 + 8, i * 64 + 3 * 16 + 8);
	for (i = 0; i < 1; i++) radix_4_fft_1(fft_array, i * 256 + 0 * 64 + 32, i * 256 + 1 * 64 + 32, i * 256 + 2 * 64 + 32, i * 256 + 3 * 64 + 32);

	for (i = 0; i < 16; i++) radix_4_fft_2(fft_array, i * 16 + 0 * 4 + 1, i * 16 + 0 * 4 + 3, i * 16 + 1 * 4 + 1, i * 16 + 1 * 4 + 3, i * 16 + 2 * 4 + 1, i * 16 + 2 * 4 + 3, i * 16 + 3 * 4 + 1, i * 16 + 3 * 4 + 3, 128, CC, angle);

	for (i = 0; i < 4; i++)
		for (j = 1; j < 8; j++)
			radix_4_fft_2(fft_array, i * 64 + 0 * 16 + j, i * 64 + 0 * 16 + 16 - j, i * 64 + 1 * 16 + j, i * 64 + 1 * 16 + 16 - j, i * 64 + 2 * 16 + j, i * 64 + 2 * 16 + 16 - j, i * 64 + 3 * 16 + j, i * 64 + 3 * 16 + 16 - j, j * 32, CC, angle);

	for (i = 0; i < 1; i++)
		for (j = 1; j < 32; j++)
			radix_4_fft_2(fft_array, i * 256 + 0 * 64 + j, i * 256 + 0 * 64 + 64 - j, i * 256 + 1 * 64 + j, i * 256 + 1 * 64 + 64 - j, i * 256 + 2 * 64 + j, i * 256 + 2 * 64 + 64 - j, i * 256 + 3 * 64 + j, i * 256 + 3 * 64 + 64 - j, j * 8, CC, angle);
}

void fft_func_512(int fft_array[], int CC, int angle[])
{
	int i;

	fft_func_256(fft_array, CC, angle);
	fft_func_256(fft_array + 256, CC, angle);
	for (i = 0; i < 1; i++) radix_2_fft_0(fft_array, 0, 256);
	for (i = 0; i < 1; i++) radix_2_fft_1(fft_array, 128, 384);
	for (i = 1; i < 128; i++) radix_2_fft_2(fft_array, 0 * 256 + i, 0 * 256 + 256 - i, 1 * 256 + i, 1 * 256 + 256 - i, i * 4, CC, angle);
}

void fft_func_1k(int fft_array[], int CC, int angle[])
{
	int i;

	fft_func_256(fft_array, CC, angle);
	fft_func_256(fft_array + 256, CC, angle);
	fft_func_256(fft_array + 512, CC, angle);
	fft_func_256(fft_array + 768, CC, angle);

	for (i = 0; i < 1; i++) radix_4_fft_0(fft_array, i * 1024 + 0 * 256, i * 1024 + 1 * 256, i * 1024 + 2 * 256, i * 1024 + 3 * 256);
	for (i = 0; i < 1; i++) radix_4_fft_1(fft_array, i * 1024 + 0 * 256 + 128, i * 1024 + 1 * 256 + 128, i * 1024 + 2 * 256 + 128, i * 1024 + 3 * 256 + 128);
	for (i = 1; i < 128; i++) radix_4_fft_2(fft_array, 0 * 256 + i, 0 * 256 + 256 - i, 1 * 256 + i, 1 * 256 + 256 - i, 2 * 256 + i, 2 * 256 + 256 - i, 3 * 256 + i, 3 * 256 + 256 - i, i * 2, CC, angle);
}

void fft_func(int fft_array[], char fft_type, char window_en, int CC, int angle[]) // fft_type 0~3:fft 256/512/1k/2k 4~7:ifft 256/512/1k/2k
{
	// if (fft_type == 0) { hanning_window(fft_array, fft_type, window_en, CC, angle); address_reverse(fft_array, 0); fft_func_256(fft_array, CC, angle); }
	// if (fft_type == 1) { hanning_window(fft_array, fft_type, window_en, CC, angle); address_reverse(fft_array, 1); fft_func_512(fft_array, CC, angle); }
	// if (fft_type == 2) { hanning_window(fft_array, fft_type, window_en, CC, angle); address_reverse(fft_array, 2); fft_func_1k(fft_array, CC, angle); }
	// if (fft_type == 3) { hanning_window(fft_array, fft_type, window_en, CC, angle); address_reverse(fft_array, 3); fft_func_2k(fft_array, CC, angle); }

	// if (fft_type == 4) { ifft_func_256(fft_array, CC, angle); address_reverse(fft_array, 4); }
	// if (fft_type == 5) { ifft_func_512(fft_array, CC, angle); address_reverse(fft_array, 5); }
	// if (fft_type == 6) { ifft_func_1k(fft_array, CC, angle); address_reverse(fft_array, 6); }
	// if (fft_type == 7) { ifft_func_2k(fft_array, CC, angle); address_reverse(fft_array, 7); }
	address_reverse(fft_array, 2); 
	fft_func_1k(fft_array, CC, angle);
}

int angle[16] = {32768, 19344, 10221,  5188,  2604, 1303, 652, 326, 163, 81, 41, 20, 10, 5, 3, 1 }; 
void get_footprint(int* fft_array, int size, float* output){
	int CC = 15;
	int adr_flag = 2;
	int i, j, k, tmp_index;
	float tempfloat;
	address_reverse(fft_array, adr_flag);
	fft_func_1k(fft_array, CC, angle);
    
    //rearrange elements positions
	input[0] = (float)fft_array[0];
	input[1] = 0.0;
	for (i = 1; i < 512; i++)
	{
		input[2 * i] = (float)fft_array[i];
		input[2 * i + 1] = (float)fft_array[1024 - i];
	}
	input[2 * 512] = fft_array[512];//0.0;
	input[2 * 512 + 1] = 0.0;

	for (i = 0; i < fft_order / 2 + 1; i++) {
		tmp_index = i * 2;
		input[i] = input[tmp_index] * input[tmp_index] + input[tmp_index + 1] * input[tmp_index + 1];	//abs & magnitude
//		printf("%f\n", input[i]);
	}
	for (j = 0; j < bandnum; j++) {
		tempfloat = 0;
		// templl = 0;
		for (k = 0; k < fft_order / 2 + 1; k++)
		{
			tempfloat += input[k] * filterCoeff[j][k];
			// templl += input[k] * i16FilterCoeff[j][k];
		}
		t_logX[j] = log(tempfloat + 0.0001);
		// templl = templl >> 15;
		// t_logX[j] = log(templl + 0.0001);
	}
	int idx;
    for (idx = 0; idx < size; ++idx){
    	printf("%d: %f",idx, t_logX[idx]);
        output[idx] = t_logX[idx];
    }
}