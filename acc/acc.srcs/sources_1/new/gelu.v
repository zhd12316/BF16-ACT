module gelu(
        input  wire        clk,
        input  wire        rst,


        // output wire [511:0]  gelu_out,
        // output wire         gelu_valid,

        //bf16_2_fp32
        input  wire [1023:0]  x_data_in, //输入32个32位浮点数
        input  wire           data_valid,  

        //mul
        // output reg  [1023:0] mul_in1_2,
        // output reg  [1023:0] mul_in2_2,
        // output reg           mul_data_valid_2,
        // input  wire [1023:0] mul_out_2,
        // input  wire          mul_valid_2,
        output reg  [1023:0] mul_out_0,
        output reg  [1023:0] mul_out_1,
        output reg           mul_valid,

        //add
        // output reg  [1023:0] add_in1,
        // output reg  [1023:0] add_in2,
        // output reg  [31:0]   add_in3,
        // output reg           add_data_valid,
        // input  wire [1023:0] add_out,
        // input  wire          add_valid,

        //exp
        // output reg  [1023:0] exp_in,    
        // output reg           exp_valid_in,
        input  wire [1023:0] exp_out,
        input  wire         exp_valid,
        output reg  [1023:0] exp_out_final_mul,
        output reg  [1023:0] exp_out_final_add,
        output reg           exp_valid_final

        //div
        // output reg  [1023:0] div_in1,
       // output reg  [31:0]   div_in2, 
        // output reg  [1023:0] div_in3,
        // output reg           div_valid_in
        // input  wire [1023:0] div_out,
        // input  wire          div_valid
);

// reg [31:0]  x1_out;
// reg         x1_valid;
// wire [3:0] mul_valid;

localparam  SQ2PIx2 = 32'h3FCC422A;//（2*根号下（2/pi））
localparam  SQ2PIMULx2 = 32'h3d922279;//（2*根号下（2/pi））乘0.044715

// reg [1:0]   state_cnt;

//占位用，防止例化端口悬空
// reg [31:0] mul_in3_0;
// reg [31:0] mul_in3_1;
// reg [31:0] mul_in3_2;
// reg [31:0] mul_in3_3;
// reg [31:0] mul_in3_4;


//第一次，前一半存SQ2PIMUL*x的结果，后一半乘x*x的结果


wire    mul_valid_0;
wire    mul_valid_1;
// wire    mul_valid_2;
// reg    mul_valid_3;
// wire    mul_valid_4;   

//=============================================
//乘法器输入信号定义    
//=============================================

// reg [1023:0] mul_in1_0;//32*32输入
// reg [1023:0] mul_in2_0;

// reg [1023:0] mul_in1_1;
// reg [1023:0] mul_in2_1;

// reg [1023:0] mul_in1_2;//32*32输入
// reg [1023:0] mul_in2_2;

// wire [1023:0] mul_in1_3;
// reg [1023:0] mul_in2_3;

wire [1023:0] mul_out_0_in;
wire [1023:0] mul_out_1_in;

// wire [1023:0] mul_out_2;
// reg [1023:0] mul_out_3;
// wire [1023:0] mul_out_4;

// reg [1023:0] mul_in1_4;
// reg [1023:0] mul_in2_4;


// reg [31:0] mul_out_temp_0;
// reg [31:0] mul_out_temp_1;

//bram例化的定义
// wire ena_x;
// wire wea_x;
// reg enb_x;
// reg [4:0] addra_x;
// wire [1023:0] dina_x;
// reg [4:0] addrb_x;
// wire [1023:0] doutb_x;


// reg     mul_data_valid_0;
// reg     mul_data_valid_1;
// reg     mul_data_valid_3;
// reg     mul_data_valid_4;

// wire[1023:0] mul_out_3_t;
// wire mul_valid_3_t;

reg [1023:0] exp_out_checked;       //这个是输入给乘法器的，得到的xexp(x1)=x
reg [1023:0] exp_out_checked_add;   //这个是输入给加法器计算exp(x1)+1的，要置为0   

reg exp_valid_t;
reg[1023:0] exp_out_t;
// reg add_flag;
// wire add_valid_negedge;
//32个乘法器阵列，64个输入端口

mul u_mul_0(
    .clk(clk),
    .rst(rst),
    .mul_in1(x_data_in),
    .mul_in2(1024'b0),
    .mul_in3(SQ2PIMULx2),
    .data_valid(data_valid),
    .mode(2'b10), 
    .mul_out(mul_out_0_in),//32*32个输出
    .mul_valid(mul_valid_0)
);

mul u_mul_1(
    .clk(clk),
    .rst(rst),
    .mul_in1(x_data_in),
    .mul_in2(x_data_in),
    .mul_in3(32'b0),
    .data_valid(data_valid),
    .mode(2'b01),
    .mul_out(mul_out_1_in),//32*32个输出
    .mul_valid(mul_valid_1)
);

// mul u_mul_3(
//     .clk(clk),
//     .rst(rst),
//     .mul_in1(mul_in1_3),
//     .mul_in2(mul_in2_3),
//     .mul_in3(mul_in3_3),
//     .data_valid(mul_data_valid_3),
//     .mode(2'b10),
//     .mul_out(mul_out_3_t),//32*32个输出
//     .mul_valid(mul_valid_3_t)
// );



//多例化一个，为了消除复用.向量乘向量
// mul u_mul_4(
//     .clk(clk),
//     .rst(rst),
//     .mul_in1(mul_in1_4),
//     .mul_in2(mul_in2_4),
//     .mul_in3(mul_in3_4),
//     .data_valid(mul_data_valid_4),
//     .mode(2'b01),
//     .mul_out(mul_out_4),//32*32个输出
//     .mul_valid(mul_valid_4)
// );


// assign ena_x = data_valid;
// assign wea_x = data_valid;

// blk_mem_gen_0 u_bram_x (
//   .clka(clk),    // input wire clka
//   .ena(ena_x),      // input wire ena
//   .wea(wea_x),      // input wire [0 : 0] wea
//   .addra(addra_x),  // input wire [4 : 0] addra
//   .dina(x_data_in),    // input wire [1023 : 0] dina
//   .clkb(clk),    // input wire clkb
//   .enb(enb_x),      // input wire enb
//   .addrb(addrb_x),  // input wire [4 : 0] addrb
//   .doutb(doutb_x)  // output wire [1023 : 0] doutb
// );

// always@(*)begin
//     enb_x = exp_valid ||(mul_valid_0 && mul_valid_1);
// end

//设置乘法器0，1输入数据

always @(posedge clk)begin
    mul_valid <= mul_valid_0 && mul_valid_1;
    mul_out_0 <= mul_out_0_in;
    mul_out_1 <= mul_out_1_in;
end


//将exp_out拆分成32个32位浮点数
wire [31:0] exp_out_slide[0:31];

genvar k;
generate
for(k=0;k<32;k=k+1)begin
   assign exp_out_slide[k] = exp_out[k*32 +:32];
end
endgenerate

always@(posedge clk)begin
    exp_valid_final     <= exp_valid;
    exp_out_final_mul   <= exp_out_checked;      //这里已经检查过nan
    exp_out_final_add   <= exp_out_checked_add;  //这里已经检查过nan
end
integer  j;
//检查exp输出是不是+inf
always@(*)begin
    exp_out_checked = exp_out;
    exp_out_checked_add = exp_out;
    for(j=0;j<32;j=j+1)begin
        if((exp_out_slide[j][31:23] == 9'h0FF)&&(!(|(exp_out_slide[j][22:0]))))begin
            exp_out_checked[j*32 +: 32] = 32'h3f800000;
            exp_out_checked_add[j*32 +: 32] = 32'd0;
        end
        else begin
            exp_out_checked[j*32 +: 32] = exp_out_slide[j];
            exp_out_checked_add[j*32 +: 32] = exp_out_slide[j];
        end
    end
end


// reg mul_data_valid_3_t;
// always@(posedge clk)begin 
//     mul_data_valid_3 <= mul_data_valid_3_t;
//     //mul_in1_3 <= doutb_x;
//     mul_in3_3 <= SQ2PIx2;
// end

// assign mul_in1_3 = doutb_x;


//为乘法器2.3设置输入数据
// always @(*)begin
//     mul_in1_2 = 1024'd0;
//     mul_in2_2 = 1024'd0;
//     mul_data_valid_2 = 1'b0;
//     mul_data_valid_3_t = 1'b0;
//     if((mul_valid_0 == 1'b1) && (mul_valid_1 == 1'b1)) begin
//         mul_data_valid_2 = 1'b1;
//         mul_data_valid_3_t = 1'b1;
//         mul_in1_2 = mul_out_0;
//         mul_in2_2 = mul_out_1;
//     end
// end

// always@(posedge clk)begin
//     mul_out_3 <= mul_out_3_t;
//     mul_valid_3 <= mul_valid_3_t;
// end


//==============================================
//计算exp(2y)+1
//=============================================
// reg [1023:0] add_1_in1;
// reg [1023:0] add_1_in2;
// reg [31:0]   add_1_in3;
// reg          add_1_data_valid;
// wire [1023:0] add_1_out;
// wire          add_1_valid;



// add u_add_1 (
//     .clk(clk),
//     .rst(rst),
//     .add_in1(add_1_in1),
//     .add_in2(add_1_in2),
//     .add_in3(add_1_in3),
//     .data_valid(add_1_data_valid),
//     .mode(2'b10),       //向量乘向量
//     .add_out(add_1_out),
//     .add_valid(add_1_valid)
// );
//设置加法器1输入数据
// always@(*)begin
//     add_in1 = 1024'd0;
//     add_in2 = 1024'd0;
//     add_in3 = 32'd0;
//     add_data_valid = 1'b0;
//     if(exp_valid) begin
//         add_data_valid = 1'b1;
//         add_in1 = exp_out_checked_add;
//     end
// end


//设置加法器输入数据
// always@(*)begin
//     add_in1 = 1024'd0;
//     add_in2 = 1024'd0;
//     add_in3 = 32'd0;
//     add_data_valid = 1'b0;
//     if(mul_valid_2 && mul_valid_3) begin
//         add_data_valid = 1'b1;
//         add_in1 = mul_out_2;
//         add_in2 = mul_out_3;

//     end
//     // else if(exp_valid) begin
//     //     add_data_valid = 1'b1;
//     //     add_in1 = exp_out_checked_add;
//     //     add_in2 = {32{32'h3F800000}};
//     // end
// end

//输入数据x存入bram_x
// always @(posedge clk or posedge rst) begin
//     if (rst) begin
//         addra_x <= 5'd0;
//     end
//     else if(data_valid) begin 
//         if (addra_x == 5'd23) begin
//             addra_x <= 5'd0;
//         end
//         else begin
//             addra_x <= addra_x + 1'b1;
//         end
//     end
//     else begin
//         addra_x <= addra_x;
//     end
// end 

//=============================================
//exp2x1乘x
//=============================================

// always@(*)begin
//     exp_in = 1024'd0;
//     exp_valid_in = 1'b0;
//     if(add_valid && (!add_flag)) begin
//         exp_valid_in = 1'b1;
//         exp_in = add_out;
//     end
// end
// always@(*)begin
//     exp_in = 1024'd0;
//     exp_valid_in = 1'b0;
//     if(add_valid ) begin
//         exp_valid_in = 1'b1;
//         exp_in = add_out;
//     end
// end


//读出bram_x中的x
// always @(posedge clk or posedge rst) begin
//     if (rst) begin
//   //      enb_x <= 1'b0;
//         addrb_x <= 5'd0;
//     end
//     else if(enb_x)begin
//         if(addrb_x < 23)begin                  
//             addrb_x <= addrb_x + 1; 
//  //           mul_data_in_0_b <= doutb_x;
//         end 
//         else begin
//             addrb_x <= 0;
//         end 
//     end
// end

//=============================================
//将exp2x1乘x存入bram_exp
//=============================================
// wire ena_exp;
// wire wea_exp;
// wire enb_exp;
// reg [4:0] addra_exp;
// wire [1023:0] dina_exp;
// reg [4:0] addrb_exp;
// wire [1023:0] doutb_exp;
// //将输入的exp2x1*x全部存起来，深度24，可存24*32=768个32位浮点数
// blk_mem_gen_0 u_bram_exp (
//   .clka(clk),    // input wire clka
//   .ena(ena_exp),      // input wire ena
//   .wea(wea_exp),      // input wire [0 : 0] wea
//   .addra(addra_exp),  // input wire [4 : 0] addra
//   .dina(dina_exp),    // input wire [1023 : 0] dina
//   .clkb(clk),    // input wire clkb
//   .enb(enb_exp),      // input wire enb
//   .addrb(addrb_exp),  // input wire [4 : 0] addrb
//   .doutb(doutb_exp)  // output wire [1023 : 0] doutb
// );
//bram例化的定义

// assign dina_exp = mul_out_1;


// //输入数据xexp存入bram_xexp
// assign ena_exp = mul_valid_1;
// assign wea_exp = mul_valid_1;


// always @(posedge clk or posedge rst) begin
//     if (rst) begin
//         addra_exp <= 5'd0;
//      //   dina_x <= 1024'd0;
//     end
//     else if(ena_exp) begin 
//         if (addra_exp == 5'd23) begin
//             addra_exp <= 5'd0;
//         end
//         else begin
//             addra_exp <= addra_exp + 1'b1;
//         end
//     end
//     else begin
//         addra_exp <= addra_exp;
//     end
// end 

// reg add_1_valid_t;
// reg[1023:0] add_1_out_t;
// always@(posedge clk)begin
//     add_1_valid_t <= add_1_valid;
//     add_1_out_t <= add_1_out;
// end

// //设置除法器输入数据
// always@(posedge clk)begin
//     if(add_1_valid_t) begin
//         div_valid_in <= 1'b1;
//         div_in1      <= doutb_exp;
//         div_in3      <= add_1_out_t;
//     end
//     else begin
//         div_valid_in <= 1'b0;
//     end
// end


// assign enb_exp = add_1_valid;


// //读出bram_exp中的exp
// always @(posedge clk or posedge rst) begin
//     if (rst) begin
//         addrb_exp <= 5'd0;
//     end
//     else if(enb_exp)begin
//         if(addrb_exp < 23)begin                  
//             addrb_exp <= addrb_exp + 1; 
//         end 
//         else begin
//             addrb_exp <= 0;
//         end 
//     end
// end

// //区分两次加法，在加法输入时得出这是第几次，在输出时已经体现
// always @(posedge clk or posedge rst) begin
//     if(rst)begin
//         add_flag <= 1'b0;
//     end
//     else if(add_valid_negedge)begin
//         add_flag <= 1'b1;
//     end
//     else if(data_valid)begin
//         add_flag <= 1'b0;
//     end
//     else begin
//         add_flag <= add_flag;
//     end
// end


// assign add_valid_negedge = add_valid_t & (~add_valid);



endmodule


