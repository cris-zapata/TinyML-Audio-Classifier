module nn_layer1 #(
    parameter int DATA_WIDTH = 8,
    parameter int VECTOR_SIZE = 16,
    parameter int NUM_NEURONS = 8,
    parameter int ACC_WIDTH = 20,
    parameter int QUANT_SHIFT = 7
)(
    input logic clk,
    input logic reset,
    input logic start,
    input logic signed [DATA_WIDTH-1:0] input_vector [0:VECTOR_SIZE-1],
    output logic done,
    output logic signed [DATA_WIDTH-1:0] hidden [0:NUM_NEURONS-1]
);
logic [3:0] index;
logic running;
logic start_prev;
logic launch;
assign launch = start && !start_prev && !running;
logic signed [DATA_WIDTH-1:0] current_input;
logic signed [DATA_WIDTH-1:0] neuron_weight [0:NUM_NEURONS-1];
logic signed [ACC_WIDTH-1:0]  neuron_bias   [0:NUM_NEURONS-1];
logic signed [ACC_WIDTH-1:0] accumulator [0:NUM_NEURONS-1];
logic signed [ACC_WIDTH:0] biased [0:NUM_NEURONS-1];
logic signed [ACC_WIDTH:0] relu [0:NUM_NEURONS-1];
logic signed [ACC_WIDTH:0] scaled [0:NUM_NEURONS-1];
assign current_input = input_vector[index];
always_ff @(posedge clk) begin
    if (reset)
        start_prev <= 1'b0;
    else
        start_prev <= start;
end
always_ff @(posedge clk) begin
    if (reset) begin
        index   <= 4'd0;
        running <= 1'b0;
        done    <= 1'b0;
    end
    else if (launch) begin
        index   <= 4'd0;
        running <= 1'b1;
        done    <= 1'b0;
    end
    else if (running) begin
        if (index == 4'd15) begin
            index   <= 4'd0;
            running <= 1'b0;
            done    <= 1'b1;
        end
        else begin
            index <= index + 1'b1;
            done  <= 1'b0;
        end
    end
    else begin
        done <= 1'b0;
    end
end
genvar n;
generate
    for (n = 0; n < NUM_NEURONS; n = n + 1) begin : neuron_gen
    localparam logic [2:0] NEURON_ID = n;
        layer1_params #(
            .DATA_WIDTH(DATA_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
        ) params_inst (
            .neuron(NEURON_ID),
            .input_index(index),
            .weight(neuron_weight[n]),
            .bias(neuron_bias[n])
        );
        always_ff @(posedge clk) begin
        if (reset)
        accumulator[n] <= '0;
        else if (launch)
        accumulator[n] <= '0;
        else if (running)
        accumulator[n] <= accumulator[n] + (current_input * neuron_weight[n]);
        end
        assign biased[n] ={accumulator[n][ACC_WIDTH-1], accumulator[n]} + {neuron_bias[n][ACC_WIDTH-1], neuron_bias[n]};
        assign relu[n] = biased[n][ACC_WIDTH] ? '0: biased[n];
        assign scaled[n] = relu[n] >>> QUANT_SHIFT;
        assign hidden[n] = (scaled[n] > 127) ? 8'sd127 : scaled[n][7:0];
    end
endgenerate
endmodule