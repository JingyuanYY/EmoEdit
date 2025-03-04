function _chart(d3,data)
{
  const width = 1024;
  const height = width;
  const cx = width * 0.5; // adjust as needed to fit
  const cy = height * 0.5; // adjust as needed to fit
  const radius = Math.min(width, height) / 2 - 260;
  console.log(width, height, radius)

  const tree = d3.cluster()
      .size([2 * Math.PI, radius])
      .separation((a, b) => (a.parent == b.parent ? 1 : 2) / a.depth);

  const root = tree(d3.hierarchy(data)
    .sort((a, b) => d3.ascending(a.data.name, b.data.name)));
  
  const colorMap = {
    "Object": "#ffc400",
    "Scene": "#0077e6",
    "Action": "#74E291",
    "Facial Expression": "#211C6A",
  };

  const svg = d3.create("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [-cx, -cy, width, height])
      .attr("style", "width: 100%; height: auto; font: 18px sans-serif;");

  // Append links.
  svg.append("g")
      .attr("fill", "none")
      .attr("stroke", "#20bbfc")
      .attr("stroke-opacity", 0.4)
      .attr("stroke-width", 1.5)
    .selectAll()
    .data(root.links())
    .join("path")
    .attr("stroke", d => {
        return colorMap[d.source.data.name] || "black";
      })
      .attr("d", d3.linkRadial()
          .angle(d => d.x)
          .radius(d => d.y));

  svg.append("g")
    .selectAll()
    .data(root.descendants())
    .join("circle")
    .attr("transform", d => `rotate(${d.x * 180 / Math.PI - 90}) translate(${d.y},0)`)
    .attr("fill", d => d.children ? "#555" : "#999")
      .attr("r", 2.5);

  svg.append("g")
    .attr("stroke-linejoin", "round")
    .attr("stroke-width", 3)
    .selectAll()
    .data(root.descendants())
    .join("text")
    .attr("transform", d => {
      if (d.parent === null) {
        return `translate(${d.y},0) rotate(0)`;
      }
      if (d.children && d.parent) {
        console.log(d.data.name)
        console.log([d.y,d.x])
        return `rotate(${d.x >= Math.PI ? 180 : 0}) translate(${d.y},0)  rotate(${d.x >= Math.PI ? 180 : 0})`;
      }
      return `rotate(${d.x * 180 / Math.PI - 90}) translate(${d.y},0) rotate(${d.x >= Math.PI ? 180 : 0})`;
    })
    .attr("dy", "0.31em")
    .attr("x", d => {
      if (d.parent === null) {
        return 0;
      }
      return d.x < Math.PI === !d.children ? 6 : -6;
    })
    .attr("text-anchor", d => {
      if (d.parent === null) {
        return "middle";
      }
      return d.x < Math.PI === !d.children ? "start" : "end";
    })
    .attr("paint-order", "stroke")
    .attr("stroke", "white")
    .attr("fill", "#555")
    .attr("font-weight", d => {
      return d.parent === null ? "bold" : "normal";
    })
    .attr("font-size", d => {
      return d.parent === null ? 30 : "normal";
    })
    .text(d => d.data.name);

  return svg.node();
}


function _data(FileAttachment){return(
FileAttachment("flare-2.json").json()
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  function toString() { return this.url; }
  const fileAttachments = new Map([
    ["flare-2.json", {url: new URL("./files/record_contentment.json", import.meta.url), mimeType: "application/json", toString}]
  ]);
  main.builtin("FileAttachment", runtime.fileAttachments(name => fileAttachments.get(name)));
  main.variable(observer("chart")).define("chart", ["d3","data"], _chart);
  main.variable(observer("data")).define("data", ["FileAttachment"], _data);
  return main;
}
