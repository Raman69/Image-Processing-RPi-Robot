import numpy as np
import cv2


#Raw video in
img = cv2.VideoCapture("Robot Code\Lane Detection\Assets\Lanes.mp4")

while True :
    # Setting basic parameters
    _, frame = img.read()
    frame = cv2.resize(frame, (600, 338))
    size = frame.shape[::-1][1:]
    width = int(img.get(3))
    height = int(img.get(4))
    
    #Isolating lanes
    #Converting the video frame from BGR (blue, green, red) color space to HLS (hue, saturation, lightness)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    #Performing Sobel edge detection on the L (lightness) channel of the image
    _, sobel = cv2.threshold(hls[:, :, 1], 120, 255, cv2.THRESH_BINARY)
    sobel = cv2.GaussianBlur(sobel, (3, 3), 0)
    xy = np.sqrt(np.absolute(cv2.Sobel(sobel, cv2.CV_64F, 0, 1, 3)) ** 2 + np.absolute(cv2.Sobel(sobel, cv2.CV_64F, 0, 1, 3)) ** 2)
    binary = np.ones_like(xy) 
    binary[(xy >= 110) & (xy <= 255)] = 0
    #Performing binary thresholding on the S (saturation) channel of the video frame. 
    _, s_binary = cv2.threshold(hls[:, :, 2], 70, 255, cv2.THRESH_BINARY)
    #Performing binary thresholding on the R (red) channel of the original BGR video frame. 
    _, r_binary = cv2.threshold(frame[:, :, 2], 120, 255, cv2.THRESH_BINARY)
    #Performing a bitwise AND operation to reduce noise in the image
    rs_binary = cv2.bitwise_and(s_binary, r_binary)
    thresh = cv2.bitwise_or(rs_binary, binary.astype(np.uint8))

    #Defining and plotting a region of intrest
    roi_points = np.float32(
        [
            (245, 184),  # Top-left corner
            (0, 300),  # Bottom-left corner
            (575, 337),  # Bottom-right corner
            (371, 184),  # Top-right corner
        ]
    )
    frame = cv2.polylines(frame, np.int32([roi_points]), True, (147, 20, 255), 3)
    desired_roi_points = np.float32(
    [
        [0.25 * width, 0],  # Top-left corner
        [0.25 * width, size[1]],  # Bottom-left corner
        [size[0] - 0.25 * width, size[1],],  # Bottom-right corner
        [size[0] - 0.25 * width, 0],  # Top-right corner
    ]
    )

    #Wraping image to get a birds eye view
    trans_matrix = cv2.getPerspectiveTransform(roi_points, desired_roi_points)
    inv_trans_matrix = cv2.getPerspectiveTransform(desired_roi_points, roi_points)
    warped_frame = cv2.warpPerspective(thresh,trans_matrix, size, flags=(cv2.INTER_LINEAR),)
    _, warped_frame= cv2.threshold(warped_frame, 127, 255, cv2.THRESH_BINARY)
    warped_copy = warped_frame.copy()
    warped_plot = cv2.polylines(warped_copy, np.int32([desired_roi_points]), True, (147, 20, 255), 3,)
    warped_plot = cv2.flip(warped_plot, 1)

    roi = warped_frame[0:338, 150:450]
    blank = np.zeros((338, 600), np.uint8)
    blank[0:338, 150:450] = roi
    warped_frame = blank
    
    #Creating sliding windows
    #Retrieving histogram data
    histogram = np.sum(warped_frame[int(warped_frame.shape[0] / 2) :, :], axis=0)
    #Setting slding window parameters
    no_of_windows = 10
    margin = int((1 / 12) * width)
    minpix = int((1 / 24) * width)

    frame_sliding_window = warped_frame.copy()
    window_height = np.int(warped_frame.shape[0] / no_of_windows)
    nonzero = warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = []
    right_lane_inds = []
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    for window in range(no_of_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_frame.shape[0] - (window + 1) * window_height
        win_y_high = warped_frame.shape[0] - window * window_height
        win_xleft_low = leftx_base - margin
        win_xleft_high = leftx_base + margin
        win_xright_low = rightx_base - margin
        win_xright_high = rightx_base + margin
        cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 255), 2,)
        cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 255), 2,)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on mean position
        if len(good_left_inds) > minpix:
            leftx_base = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_base = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract the pixel coordinates for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Store left and right lane pixel indices
    left_lane_inds = (
        nonzerox
        > (
            left_fit[0] * (nonzeroy ** 2)
            + left_fit[1] * nonzeroy
            + left_fit[2]
            - margin
        )
    ) & (
        nonzerox
        < (
            left_fit[0] * (nonzeroy ** 2)
            + left_fit[1] * nonzeroy
            + left_fit[2]
            + margin
        )
    )
    right_lane_inds = (
        nonzerox
        > (
            right_fit[0] * (nonzeroy ** 2)
            + right_fit[1] * nonzeroy
            + right_fit[2]
            - margin
        )
    ) & (
        nonzerox
        < (
            right_fit[0] * (nonzeroy ** 2)
            + right_fit[1] * nonzeroy
            + right_fit[2]
            + margin
        )
    )
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(
        0, warped_frame.shape[0] - 1, warped_frame.shape[0]
    )
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    #Adding lines to the image
    # Generate an image to draw the lane lines on
    warp_zero = np.zeros_like(warped_frame).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]
    )
    pts = np.hstack((pts_left, pts_right))
    # Draw lane on the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective
    # matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp,
        inv_trans_matrix,
        (frame.shape[1], frame.shape[0]),
    )
    # Combine the result with the original image
    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    frame_with_lines = result

    # Pixel parameters for x and y dimensions
    YM_PER_PIX = 10.0 / 1000  # meters per pixel in y dimension
    XM_PER_PIX = 3.7 / 781
    # Set the y-value where we want to calculate the road curvature.
    # Select the maximum y-value, which is the bottom of the frame.
    y_eval = np.max(ploty)
        # Fit polynomial curves to the real world environment
    left_fit_cr = np.polyfit(
        lefty * YM_PER_PIX, leftx * (XM_PER_PIX), 2
    )
    right_fit_cr = np.polyfit(
        righty * YM_PER_PIX, rightx * (XM_PER_PIX), 2
    )
    # Calculate the radii of curvature
    left_curvem = ((1 + (2 * left_fit_cr[0] * y_eval * YM_PER_PIX + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curvem = ((1 + (2 * right_fit_cr[0] * y_eval * YM_PER_PIX + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    # Get position of car in centimeters
    car_location = frame.shape[1] / 2
    # Fine the x coordinate of the lane line bottom
    height = frame.shape[0]
    bottom_left = (left_fit[0] * height ** 2 + left_fit[1] * height + left_fit[2])
    bottom_right = (right_fit[0] * height ** 2 + right_fit[1] * height + right_fit[2])
    center_lane = (bottom_right - bottom_left) / 2 + bottom_left
    center_offset = ((np.abs(car_location) - np.abs(center_lane)) * XM_PER_PIX * 100)

    cv2.putText(
        frame_with_lines,
        "Curve Radius: "
        + str((left_curvem + right_curvem) / 2)[:7]
        + " m",
        (0,25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_with_lines,
        "Center Offset: " + str(center_offset)[:7] + " cm",
        (0,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Final", frame_with_lines)
    cv2.imshow("Warped", warped_frame)
    cv2.imshow("Raw Image", frame)
    cv2.imshow("Threshold", thresh)
    if cv2.waitKey(1) == ord("q"):
        break

img.release()
cv2.destroyAllWindows