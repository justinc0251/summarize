// Get references to the slider and display elements
var slider = document.getElementById("sentence_count");
var display = document.getElementById("sentence_count_display");

// Update the displayed value when the slider is moved
slider.oninput = function () {
  display.innerText = this.value;  // Changed from innerHTML to innerText
};
