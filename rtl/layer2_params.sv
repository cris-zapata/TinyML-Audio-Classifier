module layer2_params #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 19
)(
    input  logic [1:0] neuron,
    input  logic [2:0] input_index,
    output logic signed [DATA_WIDTH-1:0] weight,
    output logic signed [ACC_WIDTH-1:0]  bias
);
always_comb begin
    weight = '0;
    bias   = '0;
    case (neuron)
        2'd0: begin
            bias = -19'sd1682;
            case (input_index)
                3'd0: weight = 8'sd0;
                3'd1: weight = 8'sd57;
                3'd2: weight = 8'sd0;
                3'd3: weight = -8'sd108;
                3'd4: weight = 8'sd0;
                3'd5: weight = 8'sd0;
                3'd6: weight = -8'sd67;
                3'd7: weight = 8'sd0;
                default: weight = '0;
            endcase
        end
        2'd1: begin
            bias = 19'sd2191;
            case (input_index)
                3'd0: weight = 8'sd0;
                3'd1: weight = -8'sd71;
                3'd2: weight = 8'sd0;
                3'd3: weight = -8'sd52;
                3'd4: weight = 8'sd0;
                3'd5: weight = 8'sd0;
                3'd6: weight = 8'sd26;
                3'd7: weight = 8'sd0;
                default: weight = '0;
            endcase
        end
        2'd2: begin
            bias = -19'sd1167;
            case (input_index)
                3'd0: weight = 8'sd0;
                3'd1: weight = -8'sd2;
                3'd2: weight = 8'sd0;
                3'd3: weight = 8'sd101;
                3'd4: weight = 8'sd0;
                3'd5: weight = 8'sd0;
                3'd6: weight = -8'sd31;
                3'd7: weight = 8'sd0;
                default: weight = '0;
            endcase
        end
        2'd3: begin
            bias = -19'sd1399;
            case (input_index)
                3'd0: weight = 8'sd0;
                3'd1: weight = 8'sd0;
                3'd2: weight = 8'sd0;
                3'd3: weight = -8'sd37;
                3'd4: weight = 8'sd0;
                3'd5: weight = 8'sd0;
                3'd6: weight = 8'sd41;
                3'd7: weight = 8'sd0;
                default: weight = '0;
            endcase
        end
        default: begin
            weight = '0;
            bias   = '0;
        end
    endcase
end
endmodule