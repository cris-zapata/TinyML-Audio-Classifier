module nn_layer2 #(
    parameter int DATA_WIDTH = 8,
    parameter int VECTOR_SIZE = 8,
    parameter int NUM_NEURONS = 4,
    parameter int ACC_WIDTH = 19
)(
    input logic clk,
    input logic reset,
    input logic start,
    input logic signed [DATA_WIDTH-1:0]
        hidden [0:VECTOR_SIZE-1],
    output logic done,
    output logic signed [ACC_WIDTH:0]
        logits [0:NUM_NEURONS-1]
);
logic [2:0] index;
logic running;
logic launch;
assign launch = start && !running;
logic signed [DATA_WIDTH-1:0] current_input;
logic signed [DATA_WIDTH-1:0]
    neuron_weight [0:NUM_NEURONS-1];
logic signed [ACC_WIDTH-1:0]
    neuron_bias [0:NUM_NEURONS-1];
logic signed [ACC_WIDTH-1:0]
    accumulator [0:NUM_NEURONS-1];
assign current_input = hidden[index];
always_ff @(posedge clk) begin
    if (reset) begin
        index <= 3'd0;
        running <= 1'b0;
        done <= 1'b0;
    end
    else if (launch) begin
        index <= 3'd0;
        running <= 1'b1;
        done <= 1'b0;
    end
    else if (running) begin
        if (index == 3'd7) begin
            index <= 3'd0;
            running <= 1'b0;
            done <= 1'b1;
        end
        else begin 
            index <= index + 1'b1;
            done <= 1'b0;
        end
    end
    else begin
        done <= 1'b0;
    end
end
genvar n;
generate
    for (n = 0; n < NUM_NEURONS; n = n + 1) begin : neuron_gen
        localparam logic [1:0] NEURON_ID = n;
        layer2_params #(
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
        assign logits[n] =
            {accumulator[n][ACC_WIDTH-1], accumulator[n]} + {neuron_bias[n][ACC_WIDTH-1], neuron_bias[n]};
    end
endgenerate
endmodule