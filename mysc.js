function redto1(){
    location.href="Tips.html";
}
function redto2(){
    location.href="Calc.html";
}
function gotolearn(){
    location.href="https://en.wikipedia.org/wiki/Water_footprint";
}
function myf(){
    document.getElementById("dem1").innerHTML="<h4>Phone No's:</h4><b>Mahathi</b>-9381740919<br><b>Sanjana</b>-8977893103<br><b>Sashidhar</b>-9392440588<br><b>Bharath</b>-9705293688<br><b>Nithin</b>-9059460274<br><b>Pratheek</b>-8076092621";
}
function myins(){
    document.getElementById("dem2").innerHTML="<h4>Instagram Id's:</h4><b>Mahathi</b>-ma_thi04<br><b>Sanjana</b>-sanjana__ankam<br><b>Sashidhar</b>-_.sashi_._<br><b>Bharath</b>-_bhharath<br><b>Nithin</b>-_the_nithin_<br><b>Pratheek</b>-pratheekrs";
}

function getValue() {
    var inputElement = document.getElementById("inputField");
    var inputValue = inputElement.value.toLowerCase();
    var displayElement = document.getElementById("displayValue");
    if (foodItems.hasOwnProperty(inputValue)) {
        var foodItem = foodItems[inputValue];
        displayElement.textContent = "Water Footprint (wft) of " + inputValue + ": " + foodItem.wft + " " + foodItem.unit;
    } else {
        displayElement.textContent = "invalid";
    }
}
