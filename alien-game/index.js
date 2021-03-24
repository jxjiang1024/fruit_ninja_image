let fingerLookupIndices = {
  thumb: [0, 1, 2, 3, 4],
  indexFinger: [0, 5, 6, 7, 8],
  middleFinger: [0, 9, 10, 11, 12],
  ringFinger: [0, 13, 14, 15, 16],
  pinky: [0, 17, 18, 19, 20]};

// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.

function drawPoint(y, x, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}

function drawKeypoints(keypoints) {
  const keypointsArray = keypoints;

  for (let i = 0; i < keypointsArray.length; i++) {
    const y = keypointsArray[i][0];
    const x = keypointsArray[i][1];
    drawPoint(x - 2, y - 2, 3);
  }

  const fingers = Object.keys(fingerLookupIndices);
  for (let i = 0; i < fingers.length; i++) {
    const finger = fingers[i];
    const points = fingerLookupIndices[finger].map(idx => keypoints[idx]);
    drawPath(points, false);
  }
}

function drawPath(points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model;
let ctx;
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");

async function predict() {    
  //Draw the frames obtained from video stream on a canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
   
  //Predict landmarks in hand (3D coordinates) in the frame of a video
  const predictions = await model.estimateHands(video);
  if(predictions.length > 0) {
      const result = predictions[0].landmarks;
      drawKeypoints(result, predictions[0].annotations);
  }

  requestAnimationFrame(predict);
}

async function main() {
  //Load the Handpose model
  model = await handpose.load();

  //Start the video stream, assign it to the video element and play it
  if(navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({video: true})
          .then(stream => {
              //assign the video stream to the video element
              video.srcObject = stream;
              //start playing the video
              video.play();
          })
          .catch(e => {
              console.log("Error Occurred in getting the video stream");
          });
  }

  video.onloadedmetadata = () => {
      //Get the 2D graphics context from the canvas element
      ctx = canvas.getContext('2d');
      //Reset the point (0,0) to a given point
      ctx.translate(canvas.width, 0);
      //Flip the context horizontally
      ctx.scale(-1, 1);

      //Start the prediction indefinitely on the video stream
      requestAnimationFrame(predict);
  };   
}
main();

