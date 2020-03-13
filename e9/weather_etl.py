import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('weather ETL').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.4+
#Professor said 2.3 is ok so I changed from 2.4 to 2.3

observation_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.StringType()),
    types.StructField('observation', types.StringType()),
    types.StructField('value', types.IntegerType()),
    types.StructField('mflag', types.StringType()),
    types.StructField('qflag', types.StringType()),
    types.StructField('sflag', types.StringType()),
    types.StructField('obstime', types.StringType()),
])


def main(in_directory, out_directory):

	#Read the input directory of .csv.gz files. 
    weather = spark.read.csv(in_directory, schema=observation_schema)
    # TODO: finish here.

	#field qflag (quality flag) is null
    weather = weather.filter(weather['qflag'].isNull())

	#the station starts with 'CA'
    weather = weather.filter(weather['station'].startswith('CA'))

	#the observation is 'TMAX'
    weather = weather.filter(weather['observation'] == ('TMAX'))

	#Divide the temperature by 10 so it's actually in °C, and call the resulting column tmax
    weather = weather.select(weather['station'],weather['date'],(weather['value']/10).alias('tmax'))
    #weather.show()	   

	#Keep only the columns station, date, and tmax.
    cleaned_data = weather.select(weather['station'],weather['date'],weather['tmax'])
    #cleaned_data.show()
	#Write the result as a directory of JSON files GZIP compressed (in the Spark one-JSON-object-per-line way).
    cleaned_data.write.json(out_directory, compression='gzip', mode='overwrite')
    #cleaned_data.show()	

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
