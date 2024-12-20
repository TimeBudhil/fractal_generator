Due to the fact that we ran out of time before we could implement the full functionality of the project, we were unable to implement the networking functionality.

To be able to implement this functionality, we planned to go through the peer-to-peer chat lab so that we can better understand how to use networking in C. We would have also used the pthread library to create threads for the server to handle multiple clients simultaneously.

This means that our program currently only uses the GPU of one machine to perform the computations, rather than performing distributed computing across multiple machines as we planned.

If given more time, we would have implemented the networking functionality to allow for distributed computing across multiple machines. Although using one single GPU is already a lot faster than using CPU, having networking implemented would have allowed us to scale our program to handle larger image sizes and greater maximum iteration numbers easily.

This is how we would have implemented the networking functionality:
We will be able to reuse most of the code that we have already written on both the server and the client, CPU code for the server, and the GPU code for both the server and clients. The things that need to be added is:
1. how the server can render the image, send tasks to the client(s), and receive the results simultaneously using threads.
2. how the client can receive the tasks from the server, using its GPU to computer the assigned pixels, and send the results back to the server.
3. how the server can receive the results from the client(s) and combine them into the final image. 

We would have used C's POSIX sockets to implement the networking functionality. We would have used a simple protocol to send and receive data between the server and the client(s), where the messages would contain the tasks to be computed (range of the pixel coordinates to be computed and the maximum iteration number, also other information such as window size, etc.) and the results of the computations (array of uint8 values representing the brightness of the pixels).

This is the code structure that we would have used to implement the networking functionality for the *server* (high-level description):
- Create a socket
- Bind the socket to a port
- Listen for incoming connections
- Accept incoming connections
- Create a thread to handle each client
- Send tasks to the client, and use its own GPU to compute its assigned pixels
- Receive results from the client
- Combine the results into the final image
- when the parameters are changed, send the new parameters to the client(s) and wait for the results

This is the code structure that we would have used to implement the networking functionality for the *client* (high-level description):
- Create a socket
- Connect to the server
- Receive tasks from the server
- Compute the assigned pixels using the GPU
- Send the results back to the server