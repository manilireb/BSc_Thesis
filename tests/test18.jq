for $i in (12 to 60) for $j in (13 to 75) group by $k := $i mod 3, $m := $j mod 3 order by $k descending, $m descending return {"first" : $i, "second" : $j} 
