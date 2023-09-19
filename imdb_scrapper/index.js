const express = require('express');
const request = require('request');
const cheerio=require('cheerio');
const fs= require('fs');


// var title,release,rating

// var json={title:"",release:"",rating:""}

const app= express()

app.get('/scrape',(req,res)=>{
    // scraping code
    url ="https://www.imdb.com/search/title/?count=100&groups=top_1000&sort=user_rating"

    request(url,function(error,response,html){
        var $=cheerio.load(html)

        $(".lister-item-header").filter(function (){
            var data = $(this)

            title=data.children().text()

            console.log(title)
            json.title=title.trim()
        })

        // $(".lister-item-header .text-muted.genre").filter(function (){
        //     var data = $(this)

        //     title=data.children().text()
        //     attributes=data.children().text()

        //     console.log(title)
        //     console.log(this.attributes)
        //     json.title=title.trim()
        //     json.attributes=this.attributes.trim()
        // })

        
        
        // $(".text-muted").filter(function (){
        //     var data = $(this)

        //     attributes=data.children().text()

        //     console.log(attributes)
        //     json.attributes=attributes.trim()
        // })


        // fs.writeFile("output.json", JSON.stringify(json,null,4),function(err){
        //     console.log("File successfully created check your directory ")
        // })

        // res.send("Check your Directory file is created")
        
    })

})

app.listen(5000,function(){
    console.log("server is listening on port 5000")
})