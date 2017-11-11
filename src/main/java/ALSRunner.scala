
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.log4j.Logger
import org.apache.log4j.Level

object ALSRunner {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val dataDir = "C:\\dataForScalaProjects\\"
  val moviesFile = "C:\\dataForScalaProjects\\movies.dat"
  val ratingsFile = "C:\\dataForScalaProjects\\ratings.dat"
  val usersFile = "C:\\dataForScalaProjects\\users.dat"
  val personalRatingsFile = "C:\\dataForScalaProjects\\personalRatings.txt"

  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

  case class Predict(userId: Int, movieId: Int)

  case class Movie(movieId: Int, movieName: String, rating: Float)

  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    return Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  def main(args: Array[String]): Unit = {
    //Initialize SparkSession
    val sparkSession = SparkSession
      .builder()
      .appName("spark-read-csv")
      .master("local[*]")
      .getOrCreate();

    import sparkSession.implicits._

    //Load Ratings
    val ratings = sparkSession
      .read.textFile(dataDir + "ratings.dat")
      .map(parseRating)
      .toDF()

    //Load Movies
    val moviesRDD = sparkSession
      .read.textFile(dataDir + "movies.dat").map { line =>
      val fields = line.split("::")
      (fields(0).toInt, fields(1))
    }
    //Load Predict
    val predictRDD = sparkSession
      .read.textFile(dataDir + "predict.txt").map { line =>
      val fields = line.split("::")
      Predict(fields(0).toInt, fields(1).toInt)
    }

    //Load my ratings
    val myRating = sparkSession.read.textFile(dataDir + "personalRatings.txt")
      .map(parseRating)
      .toDF()

    //show the DataFrames
    ratings.show(10)
    myRating.show(10)

    val numRatings = ratings.distinct().count()
    val numUsers = ratings.select("userId").distinct().count()
    val numMovies = moviesRDD.count()

    //get movies dictionary
    val movies = moviesRDD.collect.toMap

    val ratingWithMyRats = ratings.union(myRating)

    // Split dataset into training and testing parts
    val Array(training, test) = ratingWithMyRats.randomSplit(Array(0.5, 0.5))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    //Get trained model
    val model = als.fit(training)

    //Evaluate Model Calculate RMSE
    val predictions = model.transform(test).na.drop
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)

    println(s"Root-mean-square error = $rmse")

    //Get My Predictions
    val predict = predictRDD.toDF()
    val myPredictions = model.transform(predict).na.drop

    //Show your recomendations
    val myMovies = myPredictions.map(r => Movie(r.getInt(1), movies(r.getInt(1)), r.getFloat(2))).toDF
    myMovies.show(100)
  }
}
