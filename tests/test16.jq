concat( (for $i in ("Paris", "London", "Berlin", "Bern") where $i eq "Bern" return $i ), ( let $seq:= ("France", "England", "Germany", "Switzerland") return (" " || $seq[(2 * 3 - 2)]) ))