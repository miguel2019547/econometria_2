* primero le damos formato a la base de datos
format %tm tiempo
tsset tiempo

*luego realizamos un gráfico simple
tsline indiceg 

*el primer modelo "puede ser" el ar(1)
reg indiceg l.indiceg

*para probar ESTACIONALIDAD se generan dicótomas porque en este caso
*el modelo se mide mensualmente
gen m=month(dofm(tiempo))  /*estamos generando una nueva variable */
gen m1=(m==1)
gen m2=(m==2)
gen m3=(m==3)
gen m4=(m==4)
gen m5=(m==5)
gen m6=(m==6)
gen m7=(m==7)
gen m8=(m==8)
gen m9=(m==9)
gen m10=(m==10)
gen m11=(m==11)
gen m12=(m==12)

*luego probamos el modelo con las dicótomas generadas previamente

reg indiceg l(1/12).indiceg m2-m12 if tin(2000m1,2019m7)
*los resultados significativos son los que se incluyen para correr un nuevo modelo

reg indiceg l.indiceg l2.indiceg m2 m4 m5 m6 m7 m8 m10 m12 if tin(2000m1,2019m7)
*reg indiceg l.indiceg m2 m4 m5 m7 m8 m10 m12 if tin(2000m1,2019m7)

***** otro paso necesario es probar la tendencia *****
*realizamos una regresión ar(1) pero con menos datos para probar predicciones
*corremos el modelo con 6 datos menos

reg indiceg l.indiceg if tin(2000m1,2019m1)   /*es el ar(1)*/
predict indiceghat    /* el modelo no resulta ser bueno para predicciones */

********* Entonces se puede probar la Tendencia*********

tsline indiceg indiceghat   /* en el gráfico se evidencia el ajuste */
gen ten=_n     /* se crea una variable auxiliar que luego es incrementada*/
gen ten2=ten*ten     /* es la variable al cuadrado*/
gen ten3=ten2*ten

reg indiceg ten ten2 ten3
*debemos verificar el valor p de la t-student

**** EXTRA****
*también se recomienda probar VARSOC que se basa en criterios asintóticos
*varsoc indiceg if tin(2000m1,2019m1)

*varsoc indiceg if tin(2000m1,2019m1), maxlag(12)
    
*varsoc indiceg if tin(2000m1,2019m1) si es que no es mensual
/*para identificar el número de rezagos óptimo, y lag(12) es por los meses*/
*en este caso VARSOC pide incluir 12 rezagos, entonces probamos lo siguiente:

reg indiceg ten ten2 ten3 l(1/12).indiceg
*el modelo estimado indica que la tendencia no es significativa
*no todos los rezagos son significativos, corremos otro modelo con los elegidos

reg indiceg ten l.indiceg l6.indiceg l12.indiceg
*se confirma que la tendencia no es significativa, entonces:

reg indiceg l.indiceg l6.indiceg l12.indiceg
predict indiceghat2

*luego puedes comparar la potencia de predicción de tu modelo
tsline indiceg indiceghat indiceghat2
tsline indiceg indiceghat2

***** una nuva prueba corresponde a cambios estructurales ******
***** aparece disponible desde STATA 14 *****

reg indiceg l.indiceg l6.indiceg l12.indiceg
* la hipótesis nula es NO hay cambio estructural
* la hipótesis alterna es SI hay cambio estructural
* si p>0.05 no se rechaza la hipotesis nula

estat sbsingle     /* es una prueba posestimación */
estat sbknown, break(tm(2004m4))    /*esto prueba si existe cambio el mes 6 del 2000*/
estat sbsingle, swald awald alr     /*swald si hubo CE

si verificamos que el cambio se dió en 2004m4 por ejemplo, se aplica dicótomas*/

gen camest=0    /*generar una variable auxiliar de ceros */
replace camest=1 if tiempo>tm(2004,m3)    /*desde m4 se volverá =1 */

reg indiceg camest
reg indiceg l.indiceg camest

*luego corremos el modelo incluyendo tendencia y el cambio estructural
*para eso generamos nuevas variables

gen tend=_n
gen tend2=tend*tend
gen tend3=tend2*tend
gen tendce=tend*camest
gen tendce2=tend2*camest
gen tendce3=tend3*camest

reg indiceg tend tend2 tend3 tendce tendce2 tendce3 camest l.indiceg
* el modeo no tiene tendencia ni cambio estructural
