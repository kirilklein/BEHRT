from common.spark import spark_init, read_parquet, read_txt
#from CPRD.tabel import EHR
import pyspark.sql.functions as F
from pyspark.sql import Window

from os.path import join


spark = spark_init()

config= {
    'diagnoses': join('data', 'raw' ,'example_diagnoses.csv'),  # data path for diagnoses/medication
    'demographic': join('data', 'raw', 'example_demographic.csv'),  # data path for demographic information
    'output': join('data', 'formatted' ,'example_diagnoses.csv'),  # path to save formated file
    'col_name': 'icd10'  # column name for ICD/Med code
}
# construct parquet datafames which have the same schema
diagnoses = read_parquet(spark.sqlContext, config['diagnoses']).select(['patid','eventdate',config['col_name']]).na.drop().select(['patid','eventdate', config['col_name']])
# diagnoses: patid, eventdate, code
demographic = read_parquet(spark.sqlContext, config['demographic'])
# demographic: patid, age
diagnoses = diagnoses.na.drop()
diagnoses = diagnoses.dropDuplicates()

# demographic data
demographic = demographic.select(['patid', 'dob'])
diagnoses= diagnoses.join(demographic, diagnoses.patid == demographic.patid, 'inner').drop(demographic.patid)
diagnoses = EHR(diagnoses).cal_age('eventdate', 'dob', year=False).select(['patid', 'eventdate', 'age', config['col_name'], 'dob'])
diagnoses = diagnoses.dropDuplicates()

# set age and code to string
diagnoses = EHR(diagnoses).set_col_to_str('age').set_col_to_str(config['col_name'])

# group by date
diagnoses = diagnoses.groupby(['patid', 'eventdate']).agg(F.collect_list(config['col_name']).alias(config['col_name']), F.collect_list('age').alias('age'), F.first('dob').alias('dob'))
diagnoses = EHR(diagnoses).array_add_element(config['col_name'], 'SEP')

# add extra age to fill the gap of sep
extract_age = F.udf(lambda x: x[0])
diagnoses = diagnoses.withColumn('age_temp', extract_age('age')).withColumn('age', F.concat(F.col('age'),F.array(F.col('age_temp')))).drop('age_temp')

w = Window.partitionBy('patid').orderBy('eventdate')
# sort and merge ccs and age
diagnoses = diagnoses.withColumn(config['col_name'], F.collect_list(config['col_name']).over(w)).withColumn('age', F.collect_list('age').over(w)).groupBy('patid').agg(F.max(config['col_name']).alias(config['col_name']), F.max('age').alias('age'))

diagnoses = EHR(diagnoses).array_flatten(config['col_name']).array_flatten('age')
diagnoses.write.parquet(config['output'])

# we need to construct a dataframe with code and age columns
