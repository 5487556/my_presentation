const express=require("express");
const WebSocket=require("ws").Server;
const port=process.env.PORT||80;
const server=express().listen(port,()=>{
    console.log("listening at "+port+".");
});
const wss=new WebSocket({server});
wss.on("connection",ws=>{
    ws.on("close",()=>{

    });
    ws.on("message",data=>{
        
    });
});