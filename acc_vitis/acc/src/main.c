#include "xparameters.h"
#include "xil_io.h"
#include "xil_types.h"
#include "xil_printf.h"
//#include "utils.h"

#define ACC_BASE_ADDR 0xA0000000
#define COUNT_ADDR 0x44A00000

#define RST_GPIO_BASE 0x40000000

void pl_reset()
{
    Xil_Out32(RST_GPIO_BASE, 1);
    usleep(3);

    Xil_Out32(RST_GPIO_BASE, 0);
    usleep(3);
}

int main()
{
	pl_reset();

	xil_printf("Configure function type: Softmax\r\n");
	Xil_Out32(ACC_BASE_ADDR, 0x00000000);

	xil_printf("Wait for end of compute\r\n");
	usleep(1000000);

	u32 count = Xil_In32(COUNT_ADDR);
	xil_printf("number of compute clock is: %d\r\n", count);


	pl_reset();

	xil_printf("Configure function type: LayerNorm\r\n");
	Xil_Out32(ACC_BASE_ADDR, 0x00000001);

	xil_printf("Wait for end of compute\r\n");
	usleep(1000000);

	u32 count1 = Xil_In32(COUNT_ADDR);
	xil_printf("number of compute clock is: %d\r\n", count1);


	return 0;
}
