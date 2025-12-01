//****************************** AXI *****************************************
`define AXI_DATA_WIDTH  512
`define AXI_ADDR_WIDTH  19
// M_AXIS 地址总线宽度
`define C_M_AXIS_TDATA_WIDTH 1024
// S_AXIS 数据总线宽度
`define C_S_AXIS_TDATA_WIDTH 512
// Master 在启动 transaction 前等待的时钟周期
`define C_M_START_COUNT	32
`define C_S_AXI_ADDR_WIDTH 5
`define C_S_AXI_DATA_WIDTH 32