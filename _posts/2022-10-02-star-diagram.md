---
title: "Hertzsprung–Russell Diagram in D3.js"
date: "2022-10-02"
---

A Hertzsprung–Russell diagram (HR diagram) is a visualization of star
data which shows the relationship between magnitude and spectral
characteristics. The diagram was created by Ejnar Hertzsprung an
Henry Norris Russell independently in the early 20th century. You 
can read more about these diagrams here.

While interesting, I am no astronomer and am primarily inspired by
how interesting the diagrams appear. I originally saw this diagram
on a post my Mike Bostock (creator of D3.js) when learning more about
creating data visualizations in JavaScript. You can see his 
implementation here.

My visual uses the same underlying CSV as Mike Bostock’s visual,
but simplifies the output and makes it smaller. It also detects
user scrolls to turn individual star data points on and off to
create a star-twinkle effect. The effect is most pronounced on
smooth scrolls (such as a touchscreen device or trackpad).

In all, this is more of an exercise in art than data analysis. Enjoy!

<div id="d3-chart-hz"></div>

<script>
    
//// DEFINE DATA AND AXES ////

d3.csv("/assets/star-catalog.csv", function (d) {
    return {
        color: +d.color,
        absolute_magnitude: +d.absolute_magnitude
    }
}).then(function (data) {

    data_hz = [];
    Object.entries(data).forEach(function (d) {
        if (d[0] % 10 == 0) { data_hz.push(d[1]) };
    });

    //// BEGIN SVG SETUP ////

    // Determine width of container
    var div_spec = d3.select("#d3-chart-hz").node().getBoundingClientRect();

    // Set margins for D3
    var margin = 30;
    var width = +div_spec.width - margin - margin;
    var height = 600 - margin - margin;

    // Append SVG to DOM
    var svg_hz = d3.select("#d3-chart-hz")
        .append("svg")
        .attr("width", width + margin + margin)
        .attr("height", height + margin + margin)
        .style("background", "#000")

    // Add container for elements
    var plot_hz = svg_hz.append("g")
        .attr("id", "container")
        .attr("transform", "translate(" + margin + "," + margin + ")");

    // Create x axis function
    var ext_hz = d3.extent(data_hz, (d) => d.color);
    ext_hz[0] = ext_hz[0] - 0.2;
    var x_fun_hz = d3.scaleLinear()
        .domain(ext_hz)
        .range([0, width]);

    // Create y axis function
    var y_fun_hz = d3.scaleLinear()
        .domain(d3.extent(data_hz, (d) => d.absolute_magnitude))
        .range([0, height]);

    // Create color function
    var color_fun_hz = d3.scaleSequential(d3.interpolateRdBu)
        .domain(ext_hz.reverse());

    // Append x-axis to DOM
    plot_hz.append("g")
        .attr("id", "x-axis")
        .attr("transform", "translate(0," + height + ")")
        .attr("color", "white")
        .call(d3.axisBottom(x_fun_hz))
        .selectAll(".domain")
        .remove();

    // Append y-axis to DOM
    plot_hz.append("g")
        .attr("id", "y-axis")
        .attr("color", "white")
        .call(d3.axisLeft(y_fun_hz))
        .selectAll(".domain")
        .remove();

    // Append axis label
    plot_hz.append("text")
        .text("Star Color (Blue to Red)")
        .attr("x", width / 2)
        .attr("y", height - 10)
        .style("fill", "white")
        .attr("text-anchor", "middle")
        .attr("font-size", "10px");

    // Append axis label
    plot_hz.append("text")
        .text("Absolute Magnitude (Lower is Brighter)")
        .attr("transform", "rotate(-90)")
        .attr("x", 0 - height / 2)
        .attr("y", 0 + 12)
        .style("fill", "white")
        .attr("text-anchor", "middle")
        .attr("font-size", "10px");

    // Data length
    var hz_len = data_hz.length
    var plot_hz_circles = plot_hz.append("g").attr("id", "circles")

    // Function to append points
    function plot_point_ratio() {
        plot_hz_circles.selectAll("*").remove()
        plot_hz_circles.selectAll("circle")
            .data(data_hz)
            .enter()
            .append("circle")
            .attr("cx", (d) => x_fun_hz(d.color))
            .attr("cy", (d) => y_fun_hz(d.absolute_magnitude))
            .style("fill", (d) => color_fun_hz(d.color))
            .style("opacity", (d) => Math.random())
            .attr("r", 0.75);
    }

    // Initial state
    plot_point_ratio();

    // Scroll state with 100ms throttle
    var time = Date.now();
    document.addEventListener("scroll", function () {
        if (time + 100 - Date.now() < 0) {
            plot_point_ratio();
            time = Date.now();
        }
    }, true);

});

</script>
