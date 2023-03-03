/**
 * INSTRUCTIONS:
 * Left, right, and middle click to place dots on the screen.
 * Press Space to reset the constants in the network
 * Press Up Arrow Key to toggle between gradient mode, and non-gradient mode
 * Press Down Arrow to save your network constants.
 * Press any other key to remove the last dot
 * 
 * Watch as the network tries to classify them!
**/


//The higher this is, the faster the training, but you might miss a true local min.
const LEARNING_RATE = 0.2;

// Choice of input transformation
let Φ = (x, y) => {
    return [x, y, x * x, y * y, x * y];
}

// Utility functions


var standardNorm;

function* range(...args) {
    let start, end, step;
    if (args.length === 0) return;
    else if (args.length === 1) [start, end, step] = [0, args[0], 1];
    else if (args.length === 2) [start, end, step] = [args[0], args[1], 1];
    else { [start, end, step] = args; }

    for (let i = start; i < end; i += step) yield i;
}

var subtract_list = (a, b) => a.map((a, i) => a - b[i]);
var last_of = (a) => a[a.length - 1];



let sigmoid = x => 1 / (1 + exp(-x));




// MACHINE LEARNING


/*
--------------------------
DIFFERENT KINDS OF NEURONS
--------------------------
*/

class input_neuron {
    value;
    lastValue;
    constructor(val) {
        this.value = val;
        this.lastValue = val;
    }

    evaluate = () => {
        return this.value;
    }

    weights = () => {
        return [1];
    }

    backPropagate = () => {}
}

class sigmoid_neuron {
    #weights = [];
    lastValue;
    lastInput;

    constructor(numInputs) {
        for (let i = 0; i <= numInputs; i++) { this.#weights.push(standardNorm()); }
    }

    evaluate = (inputs) => {
        this.lastInput = [new input_neuron(1)].concat(inputs);
        this.lastValue = sigmoid(this.#weights.map((val, index) => val * this.lastInput[index].lastValue).reduce((a, b) => a + b, 0));
        return this.lastValue;
    }

    weights = () => {
        return this.#weights;
    }

    backPropagate = (target) => {
        let sig_prime = this.lastValue * (1 - this.lastValue);

        for (const [index, input] of this.lastInput.entries()) {
            input.backPropagate(this.#weights[index] * sig_prime * target);
        }

        for (const index in this.#weights) {
            this.#weights[index] += LEARNING_RATE * this.lastInput[index].lastValue * sig_prime * target;
        }
    }
}

class relu_neuron {
    #weights = [];
    lastValue;
    lastInput;

    constructor(numInputs) {
        for (let i = 0; i <= numInputs; i++) { this.#weights.push(standardNorm()); }
    }

    evaluate = (inputs) => {
        this.lastInput = [new input_neuron(1)].concat(inputs);
        this.lastValue = max(0, this.#weights.map((val, index) => val * this.lastInput[index].lastValue).reduce((a, b) => a + b, 0));
        return this.lastValue;
    }

    weights = () => {
        return this.#weights;
    }

    backPropagate = (target) => {
        let multiple = this.lastValue > 0 ? 1 : 0;

        if (multiple > 0) {
            for (const [index, input] of this.lastInput.entries()) {
                input.backPropagate(this.#weights[index] * target);
            }
        }

        for (const index in this.#weights) {
            this.#weights[index] += LEARNING_RATE * this.lastInput[index].lastValue * multiple * target;
        }
    }
}

/*
-------------------------
DIFFERENT KINDS OF LAYERS
-------------------------
*/

class inner_layer {
    neurons = [];

    constructor (num_inputs, num_outputs, neuron_type) {
        for (let i of range(0, num_outputs)) {
            this.neurons.push(new neuron_type(num_inputs));
        }
    }

    backPropagate = (target) => {
        for (const [index, value] of target.entries()) {
            this.neurons[index].backPropagate(value);
        }
    }

    evaluate = (inputs) => {
        let outputs = [];
        for (const neuron of this.neurons) {
            outputs.push(neuron.evaluate(inputs));
        }
        return outputs;
    }

    getNeurons = () => {
        return this.neurons;
    }
}

class input_layer {
    values = [];

    constructor (...vals) {
        for (const value of vals) {
            this.values.push(new input_neuron(value));
        }
    }

    evaluate = (_) => {
        return this.values.map((neuron) => neuron.value);
    }

    getNeurons = () => {
        return this.values;
    }
}

/*
------------------
THE NETWORK ITSELF
------------------
*/

class neural_network {
    layers = [];
    lastOutput;

    constructor (inputs, ...layers) {
        this.layers.push(new inner_layer(inputs, layers[0][0], layers[0][1]));
        for (let i of range(1, layers.length)) {
            let [input_size, _] = layers[i - 1];
            let [output_size, neuron_type] = layers[i];
            this.layers.push(new inner_layer(input_size, output_size, neuron_type));
        }
    }

    evaluate = (input) => {
        let sequentialOutput = this.layers[0].evaluate(input.getNeurons());
        for (let i of range(1, this.layers.length)) {
            sequentialOutput = this.layers[i].evaluate(this.layers[i - 1].getNeurons());
        }

        this.lastOutput = sequentialOutput;
        return this.lastOutput;
    }

    train = (data_point) => {
        let [input, output] = data_point;

        let evaluation = this.evaluate(new input_layer(...input));
        let diff = subtract_list(output, evaluation);

        last_of(this.layers).backPropagate(diff);
    }
}



// Structure of the network
let NETWORK_ARGS = [Φ(0, 0).length, [30, sigmoid_neuron], [3, sigmoid_neuron]];


var data = [];
var gradientmode = false;

var mouseReleased = function() {

    let [w2, h2] = [width/2, height/2];

    let location = Φ((mouseX - w2)/w2, (mouseY-h2)/h2);

    if(mouseButton === LEFT) {
        data.push([location, [1, 0, 0]]);
    }

    if(mouseButton === RIGHT) {
        data.push([location, [0, 1, 0]]);
    }

    if(mouseButton === CENTER) {
        data.push([location, [0, 0, 1]]);
    }

};

var keyReleased = function() {
    if(keyCode === 38) {
        gradientmode = !gradientmode;
    }
    
    if(keyCode === 40) {
        NETWORK = new neural_network(...NETWORK_ARGS);
    }

    if(keyCode === 32) {
        data.pop();
    }
};

var maxIndex = function(arr) {
    var index = 0;
    for(var i = 1; i < arr.length; i++) {
        if(arr[i] > arr[index]) { index = i; }
    }
    return index;
};

let NETWORK;

function setup() {
    createCanvas(800, 800);
    standardNorm = () => randomGaussian(0, 1);

    NETWORK = new neural_network(...NETWORK_ARGS);
}

var draw = function() {
    background(255);

    noStroke();

    const GRID_SIZE = 0.05;

    for(var x = -1; x < 1; x += GRID_SIZE) {
        for(var y = -1; y < 1; y += GRID_SIZE) {
            let output = NETWORK.evaluate(new input_layer(...Φ(x, y)));
            if(gradientmode){
                fill(output[0] * 255, output[1] * 255, output[2] * 255);
            }
            else {
                var maxindex = maxIndex(output);

                fill(maxindex === 0 ? 255:0, maxindex === 1 ? 255:0, maxindex === 2 ? 255:0);
            }

            rect(x * width/2 + width/2, y * height/2 + height/2, width*0.05, height*0.05);
        }
    }

    stroke(0, 0, 0);
    for(const data_point of data) {
        let [location, color] = data_point;
        fill(color[0]*225, color[1]*225, color[2]*225);
        ellipse(location[0]* width/2 + width/2, location[1] * height/2 + height/2, 7, 7);
    }

    let error = 0;
    let correct_guesses = 0;
    for (const data_point of data) {
        let [input, target] = data_point;
        let output = NETWORK.evaluate(new input_layer(...input));

        let errors = subtract_list(target, output).map(off => 1/2 * off * off).reduce((a, b) => a + b, 0);
        error += errors;

        correct_guesses += maxIndex(target) === maxIndex(output) ? 1 : 0;
    }

    fill(0, 0, 0);
    textSize(20);
    text(`Error Function: ${error}`, 40, 40);
    text(`Correct Guesses: ${correct_guesses}/${data.length}`, 40, 70);

    let start_time = millis();
    while (millis() - start_time < 50) {
        for (const data_point of data) {
            NETWORK.train(data_point);
        }
    }
};
