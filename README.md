# About

This package contains three nodes to help you compute optical flow
between pairs of images, usually adjacent frames in a video, visualize
the flow, and apply the flow to another image of the same dimensions.

Most of the code is from [Deforum](https://deforum.github.io/), so
this is released under the same license (MIT).

# Nodes

## Compute optical flow

This node takes two images, prev and current, and computes the optical
flow between them using either the DIS (Dense Inverse Search) medium
or fine method, or Farneback. The images must have the same
dimensions.

## Apply optical flow

This node takes an image and applies an optical flow to it, so that
the motion matches the original image. This can be used for example to
improve consistency between video frames in a vid2vid workflow, by
applying the motion between the previous input frame and the current
one to the previous output frame before using it as input to a sampler.

## Visualize optical flow

This node takes an image and a flow and produces an image visualizing
the flow on top of the image. The image must be the same size as the
images used to compute the flow in the first place. It's up to you
whether you use the "prev", "current", or an image you intend to apply
the flow to.

