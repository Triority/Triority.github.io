#include "Arduino.h"
#include "ser_par.h"

String inString="";
int rec_state = 0;
char rec_char;
int StrtoInt;
int loc_list;

void ser_par(int *par_list[]){
  if(Serial.available()>0){
    if(rec_state==0){
      rec_char = char(Serial.read());
      if(rec_char == 'S'){
        rec_state = 1;
      }
    }
  }

  if(Serial.available()>0){
    if(rec_state==1){
      rec_char = char(Serial.read());
      if (rec_char == 'T'){
        loc_list = inString.toInt();
        rec_state = 2;
        inString="";
      }else{
        inString += rec_char;
      }
    }
  }

  if(Serial.available()>0){
    if(rec_state==2){
      rec_char = Serial.read();
      if (rec_char == '\n'){
        StrtoInt = inString.toInt();
        *par_list[loc_list] = StrtoInt;
        rec_state = 0;
        inString="";
      }else{
        inString += char(rec_char);
      }
    }
  }
}
