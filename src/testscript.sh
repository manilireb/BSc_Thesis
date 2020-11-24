#!/bin/bash


for filename in /home/manuel/6-Semester/Thesis/bsc-reberma/cpp/tests/*.jq; do

	echo	
	printf "Run Rumble on the query"
	echo
	more $filename
	echo
	cd /home/manuel/6-Semester/Thesis/rumble
	export PATH=/usr/local/bin/spark-2.4.5-bin-hadoop2.7/bin:$PATH
	spark-submit target/spark-rumble-1.5.1-jar-with-dependencies.jar --query-path file:"$filename" --print-MLIR1 yes
	echo
	cd /home/manuel/6-Semester/Thesis
	printf "query is transformed to mlir:"
	echo
	more testquery.mlir
	echo
	printf "query is transformed back to jq"
	bsc-reberma/cpp/build/jsoniqc testquery.mlir -approach1 -asFile

	more testquery.jq

	mv testquery.jq testquerySource.jq

	echo
	echo second round
	echo
	printf "Run Rumble on the backtransformation"
	echo
	cd /home/manuel/6-Semester/Thesis/rumble
	spark-submit target/spark-rumble-1.5.1-jar-with-dependencies.jar --query-path file:/home/manuel/6-Semester/Thesis/testquerySource.jq --print-MLIR1 yes
	echo
	cd /home/manuel/6-Semester/Thesis
	printf "query is transformed to mlir":
	echo
	more testquery.mlir
	echo
	printf "query is transfomred back to jq"
	bsc-reberma/cpp/build/jsoniqc testquery.mlir -approach1 -asFile

	more testquery.jq

	file1="/home/manuel/6-Semester/Thesis/testquerySource.jq"
	file2="/home/manuel/6-Semester/Thesis/testquery.jq"

	echo
	echo

	if cmp -s "$file1" "$file2"; then
    	printf "Test passed!"
    	echo
	else
    	printf "Test failed!"
    	break
	fi
    
done
