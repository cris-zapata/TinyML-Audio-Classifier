module nn_classifier (
    input logic clk,
    input logic reset,
    input logic start,
    input logic signed [7:0] input_vector [0:15],
    output logic done,
    output logic [1:0] class_id
);
logic layer1_done;
logic layer2_done;
logic signed [7:0] hidden [0:7];
logic signed [19:0] logits [0:3];
assign done = layer2_done;
nn_layer1 layer1_inst (
    .clk(clk),
    .reset(reset),
    .start(start),
    .input_vector(input_vector),
    .done(layer1_done),
    .hidden(hidden)
);
nn_layer2 layer2_inst (
    .clk(clk),
    .reset(reset),
    .start(layer1_done),
    .hidden(hidden),
    .done(layer2_done),
    .logits(logits)
);
//logits[0] = clap
//logits[1] = whistle
//logits[2] = speech
//logits[3] = snap
always_comb begin
    class_id = 2'd0;
    if ((logits[1] > logits[0]) && (logits[1] >= logits[2]) && (logits[1] >= logits[3]))
    class_id = 2'd1;
    else if ((logits[2] > logits[0]) && (logits[2] > logits[1]) && (logits[2] >= logits[3]))
    class_id = 2'd2; 
    else if ((logits[3] > logits[0]) && (logits[3] > logits[1]) && (logits[3] > logits[2]))
    class_id = 2'd3;
end
endmodule