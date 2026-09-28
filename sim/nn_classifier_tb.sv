`timescale 1ns/1ps
module nn_classifier_tb;
logic clk;
logic reset;
logic start;
logic signed [7:0] input_vector [0:15];
logic done;
logic [1:0] class_id;
integer tests_passed;
// --------------------------------------------------
// DUT
// --------------------------------------------------
nn_classifier dut (
    .clk(clk),
    .reset(reset),
    .start(start),
    .input_vector(input_vector),
    .done(done),
    .class_id(class_id)
);
// --------------------------------------------------
// 100 MHz clock
// --------------------------------------------------
initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
end
// --------------------------------------------------
// Run one inference
// --------------------------------------------------
task automatic run_test(
    input string test_name,
    input integer expected_class,
    input integer h0,
    input integer h1,
    input integer h2,
    input integer h3,
    input integer h4,
    input integer h5,
    input integer h6,
    input integer h7,
    input integer l0,
    input integer l1,
    input integer l2,
    input integer l3
);
    logic layer1_pass;
    logic layer2_pass;
    logic classifier_pass;
begin
    // Pulse start for one clock
    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;
    // Wait for complete inference
    wait(done == 1'b1);
    #1;
    $display("");
    $display("===== %s =====", test_name);
    $display("Hidden:");
    $display("%0d %0d %0d %0d %0d %0d %0d %0d",
        dut.hidden[0],
        dut.hidden[1],
        dut.hidden[2],
        dut.hidden[3],
        dut.hidden[4],
        dut.hidden[5],
        dut.hidden[6],
        dut.hidden[7]
    );
    $display("Logits:");
    $display("%0d %0d %0d %0d",
        dut.logits[0],
        dut.logits[1],
        dut.logits[2],
        dut.logits[3]
    );
    $display("Class ID: %0d", class_id);
    // Layer 1 comparison
    layer1_pass =
        (dut.hidden[0] == h0) &&
        (dut.hidden[1] == h1) &&
        (dut.hidden[2] == h2) &&
        (dut.hidden[3] == h3) &&
        (dut.hidden[4] == h4) &&
        (dut.hidden[5] == h5) &&
        (dut.hidden[6] == h6) &&
        (dut.hidden[7] == h7);
    // Layer 2 comparison
    layer2_pass =
        (dut.logits[0] == l0) &&
        (dut.logits[1] == l1) &&
        (dut.logits[2] == l2) &&
        (dut.logits[3] == l3);
    // Final classifier comparison
    classifier_pass = (class_id == expected_class);
    if (layer1_pass)
        $display("LAYER 1: PASS");
    else
        $display("LAYER 1: FAIL");
    if (layer2_pass)
        $display("LAYER 2: PASS");
    else
        $display("LAYER 2: FAIL");
    if (classifier_pass)
        $display("CLASSIFIER: PASS");
    else
        $display("CLASSIFIER: FAIL");
    if (layer1_pass && layer2_pass && classifier_pass) begin
        tests_passed = tests_passed + 1;
        $display("TEST RESULT: PASS");
    end
    else begin
        $display("TEST RESULT: FAIL");
    end
    $display("==========================");
    // Make sure done has returned low before next test
    @(negedge clk);
    wait(done == 1'b0);
end
endtask
// --------------------------------------------------
// Main regression
// --------------------------------------------------
initial begin
    reset = 1'b1;
    start = 1'b0;
    tests_passed = 0;
    // Hold reset for two clocks
    repeat (2) @(posedge clk);
    @(negedge clk);
    reset = 1'b0;
    // ==================================================
    // TEST 1: CLAP
    // ==================================================
    input_vector[0]  = 8'sd111;
    input_vector[1]  = 8'sd124;
    input_vector[2]  = 8'sd127;
    input_vector[3]  = 8'sd127;
    input_vector[4]  = 8'sd111;
    input_vector[5]  = 8'sd98;
    input_vector[6]  = 8'sd97;
    input_vector[7]  = 8'sd92;
    input_vector[8]  = 8'sd92;
    input_vector[9]  = 8'sd86;
    input_vector[10] = 8'sd81;
    input_vector[11] = 8'sd79;
    input_vector[12] = 8'sd77;
    input_vector[13] = 8'sd76;
    input_vector[14] = 8'sd72;
    input_vector[15] = 8'sd75;
    run_test(
        "TEST 1 - CLAP",
        0,
        0, 86, 0, 8, 0, 0, 10, 0,
        1686, -4071, -841, -1285
    );
    // ==================================================
    // TEST 2: WHISTLE
    // ==================================================
    input_vector[0]  = 8'sd73;
    input_vector[1]  = 8'sd38;
    input_vector[2]  = 8'sd127;
    input_vector[3]  = 8'sd46;
    input_vector[4]  = 8'sd44;
    input_vector[5]  = 8'sd51;
    input_vector[6]  = 8'sd19;
    input_vector[7]  = 8'sd20;
    input_vector[8]  = 8'sd15;
    input_vector[9]  = 8'sd4;
    input_vector[10] = 8'sd10;
    input_vector[11] = 8'sd2;
    input_vector[12] = 8'sd2;
    input_vector[13] = 8'sd0;
    input_vector[14] = 8'sd0;
    input_vector[15] = 8'sd0;
    run_test(
        "TEST 2 - WHISTLE",
        1,
        0, 0, 0, 0, 0, 0, 0, 0,
        -1682, 2191, -1167, -1399
    );
    // ==================================================
    // TEST 3: SPEECH
    // ==================================================
    input_vector[0]  = 8'sd115;
    input_vector[1]  = 8'sd127;
    input_vector[2]  = 8'sd93;
    input_vector[3]  = 8'sd101;
    input_vector[4]  = 8'sd96;
    input_vector[5]  = 8'sd83;
    input_vector[6]  = 8'sd79;
    input_vector[7]  = 8'sd105;
    input_vector[8]  = 8'sd63;
    input_vector[9]  = 8'sd56;
    input_vector[10] = 8'sd27;
    input_vector[11] = 8'sd45;
    input_vector[12] = 8'sd52;
    input_vector[13] = 8'sd47;
    input_vector[14] = 8'sd26;
    input_vector[15] = 8'sd30;
    run_test(
        "TEST 3 - SPEECH",
        2,
        0, 40, 0, 37, 0, 0, 0, 0,
        -3398, -2573, 2490, -2768
    );
    // ==================================================
    // TEST 4: SNAP
    // ==================================================
    input_vector[0]  = 8'sd72;
    input_vector[1]  = 8'sd83;
    input_vector[2]  = 8'sd92;
    input_vector[3]  = 8'sd104;
    input_vector[4]  = 8'sd116;
    input_vector[5]  = 8'sd127;
    input_vector[6]  = 8'sd112;
    input_vector[7]  = 8'sd105;
    input_vector[8]  = 8'sd107;
    input_vector[9]  = 8'sd105;
    input_vector[10] = 8'sd108;
    input_vector[11] = 8'sd106;
    input_vector[12] = 8'sd106;
    input_vector[13] = 8'sd95;
    input_vector[14] = 8'sd91;
    input_vector[15] = 8'sd85;
    run_test(
        "TEST 4 - SNAP",
        3,
        0, 84, 0, 0, 0, 0, 99, 0,
        -3527, -1199, -4404, 2660
    );
    // ==================================================
    // Final result
    // ==================================================
    $display("");
    $display("================================");
    $display("       REGRESSION SUMMARY");
    $display("================================");
    $display("Tests passed: %0d / 4", tests_passed);
    if (tests_passed == 4)
        $display("ALL TESTS PASSED");
    else
        $display("REGRESSION FAILED");
    $display("================================");
    $display("");
    $finish;
end
// --------------------------------------------------
// Global timeout
// --------------------------------------------------
initial begin
    #5000;
    $display("");
    $display("ERROR: Simulation timed out.");
    $display("Layer 1 done = %b", dut.layer1_done);
    $display("Layer 2 done = %b", dut.layer2_done);
    $finish;
end
endmodule