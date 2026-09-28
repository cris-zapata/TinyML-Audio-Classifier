module layer1_params #(
  parameter int DATA_WIDTH = 8,
  parameter int ACC_WIDTH = 20  
)(
    input logic [2:0] neuron,
    input logic [3:0] input_index,
    output logic signed [DATA_WIDTH-1:0] weight,
    output logic signed [ACC_WIDTH-1:0] bias
);
always_comb begin
    weight = '0;
    bias = '0;
    case (neuron)
    //neuron 0
    3'd0: begin
        bias = -20'sd2004;
    end
    //neuron 1
    3'd1: begin
        bias = -20'sd9363;
        case (input_index)
        4'd0: weight = 8'sd30;
        4'd1: weight = 8'sd36;
        4'd2: weight = 8'sd10;
        4'd3: weight = -8'sd7;
        4'd4: weight = 8'sd16;
        4'd5: weight = 8'sd23;
        4'd6: weight = 8'sd9;
        4'd7: weight = -8'sd14;
        4'd8: weight = -8'sd7;
        4'd9: weight = 8'sd2;
        4'd10: weight = 8'sd10;
        4'd11: weight = 8'sd7;
        4'd12: weight = 8'sd7;
        4'd13: weight = 8'sd9;
        4'd14: weight = 8'sd31;
        4'd15: weight = 8'sd57;
        default: weight = '0;
    endcase
    end
 //neuron 2
    3'd2: begin
        bias = -20'sd336;
    end
    //neuron 3
    3'd3: begin
        bias = 20'sd854;
        case (input_index)
            4'd0:  weight = 8'sd65;
            4'd1:  weight = 8'sd43;
            4'd2:  weight = -8'sd60;
            4'd3:  weight = -8'sd32;
            4'd4:  weight = -8'sd29;
            4'd5:  weight = -8'sd18;
            4'd6:  weight = 8'sd13;
            4'd7:  weight = 8'sd20;
            4'd8:  weight = 8'sd28;
            4'd9:  weight = -8'sd2;
            4'd10: weight = -8'sd17;
            4'd11: weight = -8'sd7;
            4'd12: weight = -8'sd10;
            4'd13: weight = 8'sd4;
            4'd14: weight = 8'sd14;
            4'd15: weight = 8'sd3;
            default: weight = '0;
        endcase
    end
    //neuron 4
    3'd4: begin
        bias = -20'sd1545;
    end
    //neuron 5
    3'd5: begin
        bias = -20'sd660;
    end
    //neuron 6
    3'd6: begin
        bias = 20'sd1597;
        case (input_index)
            4'd0:  weight = -8'sd20;
            4'd1:  weight = -8'sd70;
            4'd2:  weight = -8'sd58;
            4'd3:  weight = -8'sd34;
            4'd4:  weight = 8'sd42;
            4'd5:  weight = 8'sd18;
            4'd6:  weight = 8'sd13;
            4'd7:  weight = 8'sd7;
            4'd8:  weight = 8'sd17;
            4'd9:  weight = 8'sd14;
            4'd10: weight = 8'sd33;
            4'd11: weight = 8'sd35;
            4'd12: weight = 8'sd18;
            4'd13: weight = 8'sd15;
            4'd14: weight = 8'sd26;
            4'd15: weight = 8'sd19;
            default: weight = '0;
        endcase
    end
    //neuron 7
    3'd7: begin
        bias = -20'sd719;
    end
    default: begin
        weight = '0;
        bias   = '0;
    end
endcase
end
endmodule